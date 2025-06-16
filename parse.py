
def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure',
                        help='only for cora/amazon/coauthor datasets')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=200)

    # model network
    parser.add_argument('--method', type=str, default='msp') #choices=['msp', 'gnnsafe', 'knn','bc']
    parser.add_argument('--backbone', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for GNN classifiers')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for sgc')

    # gnnsafe hyper
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--use_reg', action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
    parser.add_argument('--lamda2', type=float, default=0.1, help='weight for regularization')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--use_d_prop', action='store_true', help='whether to use dirichlet energy propagation')
    parser.add_argument('--eval_deng', action='store_true', help='--')

    parser.add_argument('--msd_threshold', type=float, default=0.5, help='threshold for msd baseline')
    parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--K_d', type=int, default=2, help='number of layers for dirichlet energy propagation')
    parser.add_argument('--alpha2', type=float, default=0.5, help='weight for residual connection in dirichlet energy propagation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
    parser.add_argument('--graph_wise', action='store_true', help='whether to use graph dirichlet energy')
    # baseline hyper
    parser.add_argument('--noise', type=float, default=0., help='param for baseline ODIN and Mahalanobis')

    # training
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')
    # detect
    parser.add_argument('--deng_level', type=float, default=1e-3, help='level of dirichlet energy')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true',
                        help='print args for hyper-parameter searching')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])


    parser.add_argument('--knn', type=int,
                        default=50, help='knn num')

    parser.add_argument('--test_all', action='store_true', help='use test all')
