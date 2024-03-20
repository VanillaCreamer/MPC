import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MPC.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei,Taobao}')
                        
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=3000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,   #common parameter
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64,64]', #common parameter
                        help='Output sizes of every layer')   
                                        
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size, 2048 for Beibei, 512 for Taobao')                    
    parser.add_argument('--lr', type=float, default=0.001,   #common parameter
                        help='Learning rate.')
      
    parser.add_argument('--n_layers_transfer', type=int, default=1,  
                        help='transfer size, 1 for beibei, 2 for taobao')
                                            
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
                        
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Gpu id')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]',
                        help='K for Top-K list')
                        

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')


    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='negative weight, [0.1,0.1,0.1] for beibei, [0.0001,0.0001,0.0001] for taobao')

    parser.add_argument('--decay', type=float, default=0.1,
                        help='Regularization, 0.1 for beibei, 10 for taobao')
                        

    parser.add_argument('--coefficient_loss', nargs='?', default='[0/20, 19/20, 1/20]',
                        help='Regularization, [0.0/20, 19/20, 1.0/20] for beibei, [1, 10000, 1] for taobao')
                        
    parser.add_argument('--coefficient_cart', nargs='?', default='[1, 1, 1]',
                        help='Regularization, [1, 1, 1] for beibei, [1, 0.1, 0.5] for taobao')
                        
    parser.add_argument('--coefficient_buy', nargs='?', default='[1, 1, 1, 1, 1]',
                        help='Regularization, [1, 1, 1, 1, 1] for beibei, [1, 0.2, 0.5, 0.2, 0.5] for taobao')
                        

    parser.add_argument('--mess_dropout', nargs='?', default='[0.3]',
                        help='Keep probability w.r.t. message dropout, 0.2 for beibei and taobao')

    return parser.parse_args()
    
