import datetime
import os


def save_result(args, accuracy):

    if args.rule == "Drichlet":
        path = './Output/{}/{}'.format(args.rule, args.Drichlet_arg)
    elif args.rule == "ill":
        path = './Output/{}/{}'.format(args.rule, args.ill_case)
    else:
        path = ''

    if not os.path.exists(path):
        os.makedirs(path)

    accuracy_file = 'accuracy_{}_{}_{}_{}.txt'.format(args.algorithm,
                                                      args.dataset,
                                                      args.model,
                                                      datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    with open(os.path.join(path, accuracy_file), 'a') as f1:
        for element in accuracy:
            f1.write(str(element))
            f1.write(' ')

    print('save finished')
    f1.close()



