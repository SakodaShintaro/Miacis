from glob import glob
from random import shuffle

class PathManager:
    def __init__(self, root_dir: str) -> None:
        self.path_list_ = glob(f"{root_dir}/dlshogi_with_gct-*.hcpe")
        shuffle(self.path_list_)
        self.index_ = 0

    def get_next_path(self) -> str:
        if self.index_ >= len(self.path_list_):
            # シャッフル
            shuffle(self.path_list_)
            self.index_ = 0

        result = self.path_list_[self.index_]
        self.index_ += 1
        return result
