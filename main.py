import torch

import utility
import model as model_module
import loss as loss_module
from option import args
from trainer import Trainer
import dataloader.dataset as dataset

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():

    if checkpoint.ok:
        loader = dataset.myData(args)
        model = model_module.Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        loss = loss_module.Loss(args, checkpoint) if not args.test_only else None
        trainer = Trainer(args, loader, model, loss, checkpoint)
        if args.test_only:
            trainer.test()
        else:
          for i in range(args.epochs):
            trainer.train()
            trainer.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
