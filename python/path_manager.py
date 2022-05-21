from glob import glob
from random import shuffle

# dlshogi_with_gct-*.hcpeとsuisho3kai-*.hcpeの同じidである棋譜ファイルへのパスをペアにして返すクラス
class PathManager:
    def __init__(self, root_dir: str) -> None:
        path_list1 = glob(f"{root_dir}/dlshogi_with_gct-*.hcpe")
        path_list1.sort()
        path_list2 = glob(f"{root_dir}/suisho3kai-*.hcpe")
        path_list2.sort()

        self.path_pair_ = list(zip(path_list1, path_list2))

        shuffle(self.path_pair_)
        self.index_ = 0

    def get_next_path(self) -> str:
        if self.index_ >= len(self.path_pair_):
            # シャッフル
            shuffle(self.path_pair_)
            self.index_ = 0

        result = self.path_pair_[self.index_]
        self.index_ += 1
        return result
