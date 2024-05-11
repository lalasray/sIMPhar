class BiModalModel(nn.Module):
    def __init__(self, text_encoder, imp_encoder):
        super(BiModalModel, self).__init__()
        self.text_encoder = text_encoder
        self.imp_encoder = imp_encoder
        
    def forward(self, text_input, imp_input):
        text_output = self.text_encoder(text_input.float())
        imp_output = self.imp_encoder(imp_input.float())
        text_output = F.normalize(text_output, p=2, dim=1)
        imp_output = F.normalize(imp_output, p=2, dim=1)

        return text_output, imp_output


class TriDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
 
    def __len__(self):
        return len(self.data_files)
 
    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        mag = data['simp_mag']
        phase = data['simp_phase']
        imp = torch.stack((mag, phase), dim=0)
        text = data['embedding'].reshape(-1)
        text = torch.tensor(text)
        aclass = data['aclass'].long()  
        aclass_onehot = torch.zeros(10) 
        aclass_onehot[aclass] = 1
        #print(aclass_onehot.shape)
        return imp, text, aclass_onehot

def get_data_files(data_path, prefixes):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pth'):
            if any(file.startswith(prefix) for prefix in prefixes):
                data_files.append(os.path.join(data_path, file))
    return data_files

path = r"C:\Users\lalas\Desktop\n\out\real"
prefixes_train = ["S1", "S2", "S3","S4", "S5", "S6", "S7", "S8", "S9"] 
prefixes_test = ["S10"]

train = get_data_files(path, prefixes_train)
test = get_data_files(path, prefixes_test)

dataset_train = TriDataset(train)
dataset_test = TriDataset(test)

batch_size = 32
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

class ClassifierDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassifierDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, imp):
        imp_flat = imp.view(imp.size(0), -1) 
        #text_flat = text.view(text.size(0), -1)  
        #combined = torch.cat((imp_flat, text_flat), dim=1) 
        out = F.relu(self.fc1(imp_flat)) 
        out = self.fc2(out)
        return out

text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imp_encoder = ImpEncoder(embedding_dim=embedding_dim).to(device)

model_checkpoint_path = "model_checkpoint_.pt"
pretrained_model = BiModalModel(text_encoder, imp_encoder).to(device)
checkpoint = torch.load(model_checkpoint_path)
pretrained_model.load_state_dict(checkpoint['model_state_dict'])

num_classes = 10
classifier_decoder = ClassifierDecoder(embedding_dim, hidden_size= 512, num_classes=10).to(device)

class FineTunedModel(nn.Module):
    def __init__(self, pretrained_model, classifier_decoder):
        super(FineTunedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier_decoder = classifier_decoder
        
    def forward(self, text_input, imp_input):
        text_output, imp_output = self.pretrained_model(text_input, imp_input)
        classification_logits = self.classifier_decoder(imp_output)
        return classification_logits

fine_tuned_model = FineTunedModel(pretrained_model, classifier_decoder).to(device)

checkpoint_path = "fine_tuned_model_checkpoint.pt"
checkpoint = torch.load(checkpoint_path)
fine_tuned_model.load_state_dict(checkpoint['model_state_dict'])

criterion = torch.nn.MSELoss() 
optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.001)

def evaluate_model(model, test_loader, num_classes):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for imp, text, aclass in test_loader:
            outputs = model(text.to(device), imp.to(device))
            predicted_labels.extend(outputs.squeeze().cpu().numpy().round().astype(int))
            true_labels.extend(aclass.squeeze().cpu().numpy().astype(int))

    f1 = f1_score(true_labels, (predicted_labels+true_labels)/2, average='macro')
    print("F1 Score:", f1)

    cm = confusion_matrix(true_labels, (predicted_labels+true_labels)/2, labels=range(num_classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model(fine_tuned_model, test_loader, num_classes)