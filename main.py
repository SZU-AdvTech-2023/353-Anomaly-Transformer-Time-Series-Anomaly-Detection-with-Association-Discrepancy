import os
import argparse

from torch.backends import cudnn
from utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    #为卷积层搜索最适合的卷积实现算法
    cudnn.benchmark = True
    #存储路径不存在则创建
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    #新建python模块，包括命令行选项、参数和子命令解析器
    #通过命令行运行python脚本时，可以通过ArgumentParser来解析命令行参数
    #创建ArgumentParser对象
    parser = argparse.ArgumentParser()

    #添加参数,type规定参数取值类型，default是默认值,使用--的name需要在传参的时候重写name
    # 如--data_path './data.csv'，没提到的则输出默认值
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    #确定引入哪个数据集
    parser.add_argument('--dataset', type=str, default='credit')
    #进行训练or测试
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    #数据集类型
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    #模块存储路径
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4)

    #解析添加的参数
    config = parser.parse_args()

    #vars返回对象的属性和属性值的字典对象
    args = vars(config)
    #打印出每个参数信息
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
