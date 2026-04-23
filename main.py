import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    # use_dgr_prior 保留向后兼容，但优先使用 dgr_mode
    parser.add_argument('--use_dgr_prior', type=str2bool, default='false')

    # 新增：明确指定 DGR 模式，避免硬编码开关
    # none       → E1，原始高斯先验（use_dgr_prior=False）
    # dynamic    → E2，DGRPrior 动态先验
    # multiscale → E3，MultiScaleDGRPrior 多尺度动态先验
    # static     → E4，StaticDGRPrior 静态可学习先验
    parser.add_argument('--dgr_mode', type=str, default='none',
                        choices=['none', 'dynamic', 'multiscale', 'static'])

    config = parser.parse_args()

    # dgr_mode 覆盖 use_dgr_prior，保证两者一致
    if config.dgr_mode != 'none':
        config.use_dgr_prior = True
    else:
        config.use_dgr_prior = False

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
