import datetime
import os


def save_result(data, ylabel, args):
    data = {'base': data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.algorithm, args.dataset, args.model, ylabel, args.epochs, args.lr,
                                                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    else:
        file = '{}_{}_{}_{}_{}_alpha_{}_lr_{}_{}.txt'.format(args.algorithm, args.dataset, args.model, ylabel, args.epochs,
                                                                args.data_alpha, args.lr, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            for item in data[label]:
                item1 = str(item)
                print(item1)
                f.write(item1)
                f.write(' ')
            f.write('\r\n')

    print('save finished')
    f.close()


def save_result1(data, ylabel, args):
    data = {'base': data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.algorithm, args.dataset, args.model, ylabel, args.epochs, args.lr,
                                                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    else:
        file = '{}_{}_{}_{}_{}_alpha_{}_lr_{}_{}.txt'.format(args.algorithm, args.dataset, args.model, ylabel, args.epochs,
                                                                args.data_alpha, args.lr, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            for item in data[label]:
                item1 = str(item)
                print(item1)
                f.write(item1)
                f.write(' ')
            f.write('\r\n')

    print('save finished')
    f.close()