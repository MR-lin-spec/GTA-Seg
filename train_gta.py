import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from gta.dataset.augmentation import generate_unsup_data
from gta.dataset.builder import get_loader
from gta.models.model_helper import ModelBuilder
from gta.utils.dist_helper import setup_distributed
from gta.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss_conf_weight,
    get_criterion,
)
from gta.utils.lr_helper import get_optimizer, get_scheduler
from gta.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="半监督语义分割")
parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
parser.add_argument("--local_rank", type=int, default=0, help="本地进程排名")
parser.add_argument("--seed", type=int, default=0, help="随机种子")
parser.add_argument("--port", default=None, type=int, help="端口号")

def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    # 设置实验路径和保存路径
    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    # CUDA设置
    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
        log_file = osp.join(cfg["exp_path"], "log/", f"{current_time}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # 创建网络模型
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg.get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # 优化器和学习率调度器
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    ## Teacher Assistant 模型
    model_ta = ModelBuilder(cfg["net"])
    modules_ta_back = [model_ta.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_ta_head = [model_ta.auxor, model_ta.decoder]
    else:
        modules_ta_head = [model_ta.decoder]

    if cfg.get("sync_bn", True):
        model_ta = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ta)

    model_ta.cuda()

    params_list_ta = []
    for module_ta in modules_ta_back:
        params_list_ta.append(
            dict(params=module_ta.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module_ta in modules_ta_head:
        params_list_ta.append(
            dict(params=module_ta.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer_ta = get_optimizer(params_list_ta, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model_ta = torch.nn.parallel.DistributedDataParallel(
        model_ta,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # 教师模型
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # 自动恢复 > 预训练
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            logger.error("No checkpoint found in '{}'".format(lastest_model))
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer, start_epoch=last_epoch
    )

    # 开始训练模型
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # 训练
        train(
            model,
            model_teacher,
            model_ta,
            optimizer,
            optimizer_ta,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
        )

        # 验证
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("开始验证")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(model, val_loader, epoch, logger)
            else:
                prec = validate(model_teacher, val_loader, epoch, logger)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    "\033[31m * 目前最佳验证结果是: {:.2f}\033[0m".format(best_prec * 100)
                )
                tb_logger.add_scalar("mIoU val", prec, epoch)

def train(
    model,
    model_teacher,
    model_ta,
    optimizer,
    optimizer_ta,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"标注数据 {len(loader_l)} 未标注数据 {len(loader_u)}, 数据不平衡!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = next(loader_l_iter)
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _ = next(loader_u_iter)
        image_u = image_u.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # 前向传播
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # 监督损失
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l) + 0 * rep.sum()
            else:
                sup_loss = sup_loss_fn(pred, label_l) + 0 * rep.sum()

            model_teacher.train()
            _ = model_teacher(image_l)
            model_ta.train()
            _ = model_ta(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # 将学生参数复制到教师模型
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

                    for t_params, s_params in zip(
                        model_ta.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # 生成伪标签
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # 应用强数据增强：cutout, cutmix, 或 classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # 前向传播
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs_ta = model_ta(image_all)
            
            pred_a_all, rep_a_all = outs_ta["pred"], outs_ta["rep"]
            pred_a_l, pred_a_u = pred_a_all[:num_labeled], pred_a_all[num_labeled:]
            if "aux_loss" in cfg["net"].keys():
                aux_ta = outs_ta["aux"]

            pred_a_l_large = F.interpolate(
                pred_a_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_a_u_large = F.interpolate(
                pred_a_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # 教师模型前向传播
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # 无监督损失
            unsup_loss = (
                compute_unsupervised_loss_conf_weight(
                    pred_a_u_large,
                    label_u_aug.clone(),
                    cfg["trainer"]["unsupervised"].get("drop_percent", 100),
                    pred_u_large_teacher.detach(),
                )
                * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            ) + 0.0 * rep_a_all.sum()

            optimizer_ta.zero_grad()
            unsup_loss.backward()
            optimizer_ta.step()

            model.eval()
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    0.999,
                )
                for t_params, s_params in zip(
                    model.named_parameters(), model_ta.named_parameters()
                ):
                    if 'module.decoder.classifier' not in t_params[0]:
                        t_params[1].data = (
                            ema_decay * t_params[1].data + (1 - ema_decay) * s_params[1].data
                        )

            model.train()

            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]

            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone()) + 0 * rep_all.sum()
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone()) + 0 * rep_all.sum()

        # 反向传播并更新参数
        optimizer.zero_grad()
        sup_loss.backward()
        optimizer.step()

        # 使用EMA更新教师模型
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # 收集不同GPU上的所有损失
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = 0.0
        con_losses.update(0.0)

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)

            # 将预测的图片写入TensorBoard
            grid_image = make_grid(image_l[:4], nrow=2, normalize=True)
            tb_logger.add_image('Train Images', grid_image, i_iter)
            grid_pred = make_grid(F.interpolate(pred_l_large[:4], scale_factor=4).max(1)[1].unsqueeze(1).float() / cfg["net"]["num_classes"], nrow=2, normalize=True)
            tb_logger.add_image('Train Predictions', grid_pred, i_iter)

def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # 获取模型教师的输出
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # 计算miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # 收集所有验证信息
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * 类别 [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * 第 {} 轮次 mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU

if __name__ == "__main__":
    main()


