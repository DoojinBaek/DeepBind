import time
import datetime
import argparse
    
def arg_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters')
    # parser = argparse.ArgumentParser(description='Set hyperparameters or you can use default hyperparameter settings defined in the hyperparameter.json file') 실험 다 끝난뒤 best hyperparameters 모아서 hyperparameter.json에 저장
    parser.add_argument('--TF', type=str, required=True, nargs=1, choices=['ARID3A', 'CTCFL', 'ELK1', 'FOXA1', 'GABPA', 'MYC', 'REST', 'SP1', 'USF1', 'ZBTB7A'], help='choose from [ARID3A, CTCFL, ELK1, FOXA1, GABPA, MYC, REST, SP1, USF1, ZBTB7A]')
    parser.add_argument('--pooling', type=str, required=True, nargs=1, choices=['maxavg', 'max'], help='choose from [maxavg, max]')
    parser.add_argument('--hidden', type=bool, required=True, help='Do you want to add a hidden layer?')
    parser.add_argument('--dropout_rate', type=float, required=True, nargs=1, choices=[0.2, 0.3, 0.4], help='choose from [0.2, 0.3, 0.4]')
    parser.add_argument('--lr', type=float, required=True, nargs=1, choices=[0.001, 0.005, 0.01], help='choose from [0.001, 0.005, 0.01]')
    parser.add_argument('--lr_scheduler', type=bool, required=True, help='Do you want to use cosine annealing scheduler?')
    parser.add_argument('--optimizer', type=str, required=True, nargs=1, choices=['SGD', 'Adam'], help='choose from [SGD, Adam]')
    args = parser.parse_args()
    return args

def main():
    start = time.time()
    args = arg_parser()
    tf = args.TF
    pool = args.pooling
    hidden = args.hidden
    dropout_rate = args.dropout_rate
    lr = args.lr
    scheduler = args.lr_scheduler
    opt = args.optimizer

    print(tf, pool, hidden, dropout_rate, lr, scheduler, opt)
    print(datetime.timedelta(seconds = (time.time()-start)))

if __name__ == '__main__':
    main()
