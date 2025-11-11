import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        # 读取文件并过滤掉空白行，避免生成像 "JPEGImages/.jpg" 这样的无效条目
        raw_lines = [ln.strip() for ln in open(d_list, "r")]
        lines = [ln for ln in raw_lines if ln]

        if "cityscapes" in d_list:
            self.list_sample = [
                [
                    line,
                    "gtFine/" + line[12:-15] + "gtFine_labelTrainIds.png",
                ]
                for line in lines
            ]
        elif "pascal" in d_list or "VOC" in d_list:
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line),
                    "SegmentationClassAug/{}.png".format(line),
                ]
                for line in lines
            ]
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        if self.num_sample == 0:
            # 为了让错误信息更友好，指出是哪一个列表文件导致为空
            logger.error(f"No valid samples found in list file: {d_list}")
            raise ValueError(
                f"No valid samples found in list file: {d_list}. The file may be empty or contain only comments/blank lines."
            )
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def __len__(self):
        return self.num_sample

class BaseUnlabeledDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)

        # 读取文件并过滤空白行与注释行（以 # 开头），允许列表为空但记录被忽略的行数
        raw_lines = [ln.rstrip('\n') for ln in open(d_list, "r")]
        stripped = [ln.strip() for ln in raw_lines]
        filtered = [ln for ln in stripped if ln and not ln.startswith("#")]
        ignored = len(stripped) - len(filtered)
        if ignored > 0:
            logger.info(f"Ignored {ignored} empty/comment lines in {d_list}")

        # 每个条目以单元素列表保存以兼容后续使用
        self.list_sample = [[line] for line in filtered]

        self.num_sample = len(self.list_sample)
        # 允许 unlabeled 列表为空（num_sample == 0），但记录警告
        if self.num_sample == 0:
            logger.warning(f"Unlabeled list {d_list} contains 0 samples (may be empty or comments only)")
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def __len__(self):
        return self.num_sample