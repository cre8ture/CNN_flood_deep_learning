import get_data as gd
import model
import numpy as np

import torch
from torch import tensor
from torchmetrics.regression import MeanSquaredError

import matplotlib.pyplot as plt



class ModelTraining:
    
    def __init__(self):
        
        # at beginning of the script
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def train_model(self, model, 
                    train_inputs:list, train_targets:list,
                    val_inputs:list, val_targets:list,
                    file_name_identifier:str,
                    epochs=10,
                    loss_upper_bound=6
                    ):
        """
        file_name_identifier: str. You can customise the name of the file
        """

        self.model = model.to(self.device)

        train_loss = []
        val_loss = []
        for epoch in range(epochs):  # loop over the dataset multiple times

            # training
            running_loss = 0.0
            for inputs, labels in zip(train_inputs, train_targets):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs).to(self.device)
                loss = model.criterion(outputs, labels).to(self.device)
                loss.backward()
                model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            train_loss.append(running_loss)
            print(f'training epoch {epoch + 1} loss: {running_loss:.3f}')

            # validation
            running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in zip(val_inputs, val_targets):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward + backward + optimize
                    outputs = model(inputs).to(self.device)
                    loss = model.criterion(outputs, labels).to(self.device)

                    # print statistics
                    running_loss += loss.item()
                    # if i % 2000 == 1999:    # print every 2000 mini-batches
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                val_loss.append(running_loss)
                print(f'validation epoch {epoch + 1} loss: {running_loss:.3f}')

                

        print('Finished Training')
        # save model
        PATH = './conv_net.pth'
        torch.save(model.state_dict(), PATH)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title("loss curve")
        epochs_range = np.arange(start=1, stop=epochs+1, step=1)
        ax.plot(epochs_range, train_loss, label="training")
        ax.plot(epochs_range, val_loss, label="validation")
        ax.set_ylim([0, loss_upper_bound])
        ax.set_xlabel("epochs")
        ax.set_ylabel("MSE loss")
        ax.legend()
        plt.savefig(f"./img/loss_curve_{file_name_identifier}.png")
        plt.close()

    def prediction(self, input, target, 
                   coodinate_extent=[315465, 319125, 228650, 232790],
                   vmin=0, vmax=12, 
                   save_folder_dir='./img/',
                   save_file_name_with_extension='test.png'):
        
        # general settings for plotting
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True
        im = plt.imread("./img/study_location_satellite.png")
        
        # to device
        input = input.to(self.device)
        target = target.to(self.device)

        # make prediction
        pred = self.model(input).to(self.device)
        print(f"pred shape:{pred.size()}")

        # reshape
        pred = pred.view(pred.size()[0], 138, 122)
        target = target.view(target.size()[0], 138, 122)

        
        fig, ax = plt.subplots()
        im = ax.imshow(im, extent=coodinate_extent) # extent=[0, 122, 0, 138])
        
        # img = ax.imshow(pred.detach().numpy()[20], alpha=0.5, cmap='coolwarm',
        #                 extent=coodinate_extent) # plot the 20th sample at current batch
        
        # mask out the water depths that are lower than 0.01m
        lowerBound = 0.01
        mask_zeros =np.ma.masked_where(lowerBound >= pred.detach().numpy()[0]
                                       , pred.detach().numpy()[0]) # plot the 1st map of this batch
        img = ax.imshow(mask_zeros, cmap='coolwarm', extent=coodinate_extent,
                        vmin=vmin, vmax=vmax) 
        
        fig.colorbar(img)

        ax.set_title("flood water depth (unit: m)")
        ax.set_xlabel("coordinate X")
        ax.set_ylabel("coordinate Y")
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=45)

        fig.savefig(save_folder_dir
                    +f'wd_{save_file_name_with_extension}')
        
        mean_squared_error = MeanSquaredError()
        mse = mean_squared_error(pred, target)
        print(f"mean square error:{mse}")

        return mse
        
    
    def batch_prediction(self, inputs:list, targets:list,
                        save_folder_dir='./img/',
                        save_file_name_with_extension='test.png'):

        i = 0
        mse_list = []
        for input, target in zip(inputs, targets):
            mse = self.prediction(input=input,
                            target=target,
                            save_file_name_with_extension=f'{i}.png')  
            # for non-shuffle testing data, i=[:24] are event 2010, i=[24:] are event 2011
            mse = mse.detach().numpy()
            mse_list.append(mse)
            i += 1

        fig, ax = plt.subplots(nrows=1, ncols=1)
        batches = np.arange(start=1, stop=i+1, step=1)
        ax.scatter(batches, mse_list)
        ax.set_ylim([0, 0.1])
        ax.set_title("Error plot")
        ax.set_xlabel("batch")
        ax.set_ylabel("mean square error")
        
        fig.savefig(save_folder_dir
                    +f'ac_{save_file_name_with_extension}')



def run(kwargs_list:list, 
        parameter_tuning=True,
        best_model=True,
        best_lr=0.01,
        best_dropout=0.3):

    # initialisation

    MT = ModelTraining()
    GD = gd.getData()
    
    # prepare shuffle data
    train_input_shuffle, train_target_shuffle,\
    val_input_shuffle, val_target_shuffle, \
    test_input_shuffle, test_target_shuffle = GD.prepare_data(shuffle=True)
    
    # prepare non shuffle data
    train_input, train_target,\
    val_input, val_target, \
    test_input, test_target = GD.prepare_data(shuffle=False)

    if parameter_tuning:

        for kwargs in kwargs_list:
            # train network using key word arguments
            DL = model.ConvNet(**kwargs)

            MT.train_model(model=DL, 
                        train_inputs=train_input_shuffle, train_targets=train_target_shuffle,
                        val_inputs=val_input_shuffle, val_targets=val_target_shuffle, 
                        epochs=10,
                        file_name_identifier=f"lr{kwargs['lr']}drop{kwargs['dropout']}")
            # MT.prediction(input=train_input_shuffle[0],
            #               target=train_target_shuffle[0]) # predict using 1st batch

    if best_model:
        # train best model
        DL = model.ConvNet(lr=best_lr, dropout=best_dropout) # change to the best model setting once you know it.
        
        MT.train_model(model=DL, 
                        train_inputs=train_input_shuffle, train_targets=train_target_shuffle,
                        val_inputs=val_input_shuffle, val_targets=val_target_shuffle, 
                        epochs=10,
                        file_name_identifier='best_model')
        
        # make prediction using the best model
        MT.batch_prediction(inputs=test_input,
                            targets=test_target
                            )



if __name__ == "__main__":

    lr_list = [0.1, 0.01, 0.001]
    dropout_list = [0.1, 0.2, 0.3]

    kwargs_list = [{
        'lr': lr,
        'dropout': dropout
    } for lr in lr_list for dropout in dropout_list]

    run(kwargs_list=kwargs_list,
        parameter_tuning=True,  # switch true or false for parameter tuning
        best_model=True,  # switch true or false for the best model training and testing
        best_lr=0.001,
        best_dropout=0.2)

        

    

