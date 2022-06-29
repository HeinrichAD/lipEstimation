from torch.autograd import Variable

""" Generic training and testing functions
"""


def train(model, data, loss_fn, optimizer, batch_size, n_epochs, use_cuda):
    """ Train `model` on `data`
    """
    epoch = 0
    model.train()
    while epoch < n_epochs:
        for (idx, (x, y)) in enumerate(data):
            x, y = Variable(x), Variable(y)
            if use_cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('[Epoch {} | {}/{}]: {:.4f}'.format(epoch,
                    idx, len(data),
                    loss.data[0]))
        epoch += 1


def _test(model, data, loss_fn, use_cuda=True):
    """ Test `model` on `data`
    """
    correct = 0  # Ajout JB
    test_loss = 0  # Ajout JB
    model.eval()
    # for x, y in dataset:
    for x, y in data:  # Ajout JB
        # data, target = data, target
        x, y = Variable(x), Variable(y)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        output = model(data)
        # test_loss += loss_fn(output, target).data[0]
        test_loss += loss_fn(output, y).data[0]

        pred = output.data.max(1)[1]
        # correct += pred.eq(target.data).sum()
        correct += pred.eq(y.data).sum()

        # test_loss /= len(dataset)
        test_loss /= len(data)
    print("Test set: Average loss: {:.3f},\
            Accuracy: {}/{} ({:.4f})\n".format(
                test_loss,
                correct,
                len(data.dataset),
                100. * correct / len(data.dataset)))
