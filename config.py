from loss import InfonceLoss,ContrastiveLoss

imp_encoder_type = "cnn" #fc #cnn #res #lstm #spatiotemporal
text_encoder_type = "res" #fc #cnn #res #spatial

batch_size = 32
embedding_dim = 2048
num_epochs = 300
patience = 10

classifer_type = "fc" #fc #cnn #res #lstm #spatiotemporal

classes = 10
num_epochs_class = 100
batch_size_class = 32

parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
#parent = "/home/lala/other/Repos/git/simu_wrist_har/"

loss = InfonceLoss() #ContrastiveLoss() #InfonceLoss()