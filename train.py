import torch
import torch.nn as nn
from tqdm.auto import tqdm
from dataloader import get_dataloaders
from models import load_pretrained_googlenet

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
num_diseases = 584

def main():
    # Initialize the model
    model = load_pretrained_googlenet(num_classes=num_diseases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() 

    # Get the dataloaders
    train_dataloader, _ = get_dataloaders('AtlasDermatigo_Augmented_Preprocessed')

    def train(dataloader, model, loss_fn, optimizer, epoch):
        model.train()
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (batch % 300) and batch :
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}")
        # Save model weights every 10 iterations after 20 epochs
        if epoch >= 20:
            torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")
            print(f"Saved model weights after epoch {epoch+1}")
            

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, epoch)
    print("Training complete")

if __name__ == "__main__":
    main()
