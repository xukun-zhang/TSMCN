from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./p2ilf-train/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='p2ilf-train', help='train, val, test, etc') #todo delete.
        self.parser.add_argument('--which_epoch', type=str, default='130', help='which epoch to load? latest or set to latest to use latest cached model') #130, 24.401%
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.is_train = False