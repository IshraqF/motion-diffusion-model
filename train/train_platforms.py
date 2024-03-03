import os
import wandb

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()

class WandBPlatform(TrainPlatform):
    def __init__(self, save_dir):
        os.environ['WANDB_API_KEY'] = "2e7352b6116bddfa1c6ed1d71713e5ecb8e0ffcc"
        wandb.login()
        self.run = wandb.init(
                project="Human-Motion-Diffusion",
                name="Unconstrained Training",

                config={
                    "epochs": 3134,
                    "steps": 1200000,
                    "learining_rate": 1e-4,
                    "batch_size": 64,
                    "dataset": "HumanML",
                }
        )
    
    def report_scalar(self, name, value, iteration, group_name=None):
        if group_name == 'Loss':
            wandb.log({name: value}, step=iteration)
        if group_name == 'Eval':
            wandb.log({name: value}, step=iteration)
        if group_name == 'Eval Unconstrained':
            wandb.log({name: value}, step=iteration)
    
    def close(self):
        wandb.finish()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


