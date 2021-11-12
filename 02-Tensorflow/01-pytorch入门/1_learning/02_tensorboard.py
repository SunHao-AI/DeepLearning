from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("../../logs")


if __name__ == '__main__':
    # test1
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('../../logs')
    for i in range(10):
        writer.add_scalar('quadratic', i ** 2, global_step=i)
        writer.add_scalar('exponential', 2 ** i, global_step=i)
    writer.close()

    # test2