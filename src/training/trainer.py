import torch
from tqdm.notebook import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device='cpu'):
     # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    
    running_loss = 0.0
    last_avg_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader)):
        A, B, delta = batch
   
        input = torch.cat((A, B), dim=1)
        
        input.to(device)
        delta.to(device)
    
        # forward pass
        output = model(input)
        loss = criterion(output, delta)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 20 == 19:
            last_avg_loss = running_loss / 20 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_avg_loss))
            running_loss = 0.0
        
    return last_avg_loss
    
def validate(model, val_loader, criterion, device='cpu'):
    running_loss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            A, B, delta = batch
            
            input = torch.cat((A, B), dim=1)
            
            input.to(device)
            delta.to(device)
            
            output = model(input)
            loss = criterion(output, delta)
            running_loss += loss.item()

    avg_vloss = running_loss / (i + 1)
    
    return avg_vloss

def train(model, train_loader, val_loader, lr, epochs, model_dir, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_vloss = float('inf')
    
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # scheduler.step()
        
        avg_vloss = validate(model, val_loader, criterion, device)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))    
  
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/model_{}.pth'.format(model_dir, epoch)
            torch.save(model.state_dict(), model_path)
