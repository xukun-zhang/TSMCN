from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    result_1_acc = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, acc_1 = model.test()
        result_1_acc = result_1_acc + acc_1
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    print("总输出结果的平均值：", result_1_acc/(i+1))
    return writer.acc


if __name__ == '__main__':
    run_test()
