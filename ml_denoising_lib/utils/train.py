from .training_function import calculate_psnr_with_peak
from .training_function import peak_to_peak_ratio
from .training_function import plot_metrics

import torch 
import torch.nn as nn
import torch.optim as optim

from grand_psu_lib.utils import filtering as filt

import os

def frequency_domain_loss(clean_signals, denoised_signals):
    clean_fft = torch.fft.fft(clean_signals)
    denoised_fft = torch.fft.fft(denoised_signals)
    loss = torch.mean((torch.abs(clean_fft) - torch.abs(denoised_fft))**2)
    return loss  

def combine_loss(denoised_signals, clean_signals, alpha = 0.5, beta = 0.5):
    time_domain_criterion = nn.MSELoss()
    time_loss = time_domain_criterion(denoised_signals, clean_signals)
    frequency_loss = frequency_domain_loss(clean_signals, denoised_signals)
    total_loss = alpha * time_loss + beta * frequency_loss
    return total_loss



def plot_psd_of_loader(train_loader, valid_loader, device,  save_folder = ""):
    """
    return
    1. training_losses
    2. validation_losses
    3. validation_psnr
    4. learning_rates
    5. validation_peak_to_peak 
    
"""
    for noisy_data, clean_data in train_loader:
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)


def train_validate(train_loader, valid_loader, model, optimizer, criterion, num_epochs, device, save_folder = "", scheduler=None):
    """
    return
    1. training_losses
    2. validation_losses
    3. validation_psnr
    4. learning_rates
    5. validation_peak_to_peak 
    
    """
    training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak = [], [], [], [], []
    with torch.set_grad_enabled(True):
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for noisy_data, clean_data in train_loader:
                noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)

                optimizer.zero_grad()
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                loss.backward()
                #print(loss.item())
                optimizer.step()
                if not (scheduler is None):  
                    scheduler.step()

                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            model.eval()
            total_valid_loss, total_psnr, total_peak_to_peak_ratio = 0, 0, 0
            for noisy_data, clean_data in valid_loader:
                clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)

                with torch.no_grad():
                    outputs = model(noisy_data)
                    loss = criterion(outputs, clean_data)
                    total_valid_loss += loss.item()
                    psnr_value = calculate_psnr_with_peak(clean_data.cpu().numpy(), outputs.cpu().numpy())
                    total_psnr += psnr_value
                    ratio = peak_to_peak_ratio(clean_data.cpu().numpy(), outputs.cpu().numpy())
                    total_peak_to_peak_ratio += ratio

            avg_valid_loss = total_valid_loss / len(valid_loader)
            validation_losses.append(avg_valid_loss)
            avg_psnr = total_psnr / len(valid_loader.dataset)
            validation_psnr.append(avg_psnr)
            avg_peak_to_peak_ratio = total_peak_to_peak_ratio / len(valid_loader)
            validation_peak_to_peak.append(avg_peak_to_peak_ratio)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation PSNR: {avg_psnr:.2f}, Validation Peak-to-Peak: {avg_peak_to_peak_ratio:.2f}, Learning Rate: {learning_rates[-1]:.6f}')
            if epoch % 100 == 0:
                torch.save(model.state_dict(), os.path.join(save_folder, 'model_epoch{}.pth'.format(epoch)))

    epochs = range(1, num_epochs + 1)
    plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates= learning_rates, validation_peak_to_peak = validation_peak_to_peak, save_folder = save_folder)
    return training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak



def train_one_epoch_loop(train_loader, valid_loader, model, optimizer, criterion, device, save_folder = "", scheduler=None):
    """
    return
    1. training_losses
    2. validation_losses
    3. validation_psnr
    4. learning_rates
    5. validation_peak_to_peak 
    
    """
#    training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak = [], [], [], [], []
    with torch.set_grad_enabled(True):
        #for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for k, (noisy_data, clean_data) in enumerate(train_loader):
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            if k % (len(train_loader)/10) == 0:
                print(loss.item())
            optimizer.step()
            if not (scheduler is None):  
                scheduler.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        #training_losses.append(avg_train_loss)

        model.eval()
        total_valid_loss, total_psnr, total_peak_to_peak_ratio = 0, 0, 0
        for noisy_data, clean_data in valid_loader:
            clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)

            with torch.no_grad():
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                total_valid_loss += loss.item()
                psnr_value = calculate_psnr_with_peak(clean_data.cpu().numpy(), outputs.cpu().numpy())
                total_psnr += psnr_value
                ratio = peak_to_peak_ratio(clean_data.cpu().numpy(), outputs.cpu().numpy())
                total_peak_to_peak_ratio += ratio

        avg_valid_loss = total_valid_loss / len(valid_loader)
        #validation_losses.append(avg_valid_loss)
        avg_psnr = total_psnr / len(valid_loader.dataset)
        #validation_psnr.append(avg_psnr)
        avg_peak_to_peak_ratio = total_peak_to_peak_ratio / len(valid_loader)
        #validation_peak_to_peak.append(avg_peak_to_peak_ratio)
        #learning_rates.append(optimizer.param_groups[0]['lr'])
        lr = optimizer.param_groups[0]['lr']
        print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation PSNR: {avg_psnr:.2f}, Validation Peak-to-Peak: {avg_peak_to_peak_ratio:.2f}, Learning Rate: {lr:.9f}')
        #if epoch % 100 == 0:
        #    torch.save(model.state_dict(), os.path.join(save_folder, 'model_epoch{}.pth'.format(epoch)))

    #epochs = range(1, num_epochs + 1)
    #plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates= learning_rates, validation_peak_to_peak = validation_peak_to_peak, save_folder = save_folder)
    return avg_train_loss, avg_valid_loss, avg_psnr, lr, avg_peak_to_peak_ratio



