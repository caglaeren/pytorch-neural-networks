#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN: Recurrent Neural Networks (Yinelemeli Sinir Ağları)
Yinelemli sinir ağları sıralı verileri, zaman serilerini işlemek için kullanılan derin öğrenme mimarisidir.

Veri setini seçeceğiz bu sefer.


"""

"""
  #Example: 3'lü paket için
  sequence : [1,2,3] #yani elimde üçlü bir paket var
  target : [4] #bir sonraki değeri tahmin etmeye çalışıyoruz

  sequence : [2,3,4] olsaydu
  target : [5] olurdu
  """

#%% 1- Veriyi oluşturalım
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#Veriyi oluşturmak için sinüs dalgası yapalım:
    #RNN’e verilecek her bir sequence’in uzunluğu (50 zaman adımı)
    #num_samples = Toplam zaman noktası sayısı
def generate_data(seq_length = 50, num_samples = 1000):
    
    X = np.linspace(0, 100, num_samples) # 0'dan başla, 100'e kadar, aradaki eşit aralıklı 1000 kadar veri noktası oluşturur
    y = np.sin(X) #X burda zaman, y= sinüs dalgası (asıl öğrenilecek sinyal)
    sequence = [] #Giriş dizilerini saklamak için
    targets = [] #Hedef değerleri saklamak için
    
    
    for i in range(len(X)-seq_length): #Ben bunu 50'li paketler halinde sequence içine atacağım ve bu 50'li pakete karşılık gelen target değeri belirleyeceğiz
        #sequence_length uzunlığuna giriş dizisi ekleyelim:    
        sequence.append(y[i :i+seq_length]) #i'den başla, i+seq_length'e kadar git. (0'dan başla 0+50. indekse kadar git) (input)

        targets.append(y[i + seq_length]) #input dizimizin hemen sonraki değerini hedef olarak belirler (input dizisinden sonra gelen yer )
        
        
        
    #2- Veriyi görselleştirelim (Data visualization)
    plt.figure(figsize=(8,4))
    plt.plot(X, y, label='sin(t)', color='b', linewidth=2)
    plt.title("Sinüs dalga grafiği")
    plt.xlabel("Zaman (radyan)")
    plt.ylabel("Genlik")
    plt.legend()
    plt.grid(True)
    plt.show()




    return np.array(sequence), np.array(targets) #sequence -> normalde diziydi bunu numpy array'e çevirdik


sequence, targets = generate_data() #sequence -> (950,50) verdi yani 50'lik paketlerimiz var toplam 950 adet. Buna karşılık gelen targets değerleri de 950 tane








# %% 3- RNN Modelini oluşturalım

class RNN(nn.Module): #nn.Module'den kalıtım aldı
    def __init__(self, input_size, hidden_size, output_size, num_layers=1): #constructor
        """
           1- RNN tanımlayacağız. İçine input_size, hidden_layer_size, number_of_layers, batch first gibi farklı parametreler alacak
           2- Çıktıyı üretebilmesi için tam bağlantılı katman (fully connected layer) yani lineer bir output layera bağlayacağız
           Herhangi bir şey return etmeyecek
           
           RNN() -> Linear (output)
        """
        
        
        super(RNN, self).__init__()   #super ile kalıtımı gerçekleştiririz. torch'u nn.Module'ünden inherit edecek
        
        # input_size: Giriş boyutudur. Zaman serisi verisi için 1 olarak belirleyebiliriz
        # hidden_size: Hidden layerda bulunacak olan hücrelerin sayısı (RNN gizli katman düğüm (cell) sayısı)
        # num_layers: RNN'in kaç katmanlı olacağıdır yani layer sayısı. default=1 yaptık
        self.rnn = nn.RNN(input_size , hidden_size, num_layers, batch_first=True) #Recurrent Neural Network'ümüzü tanımlayalım. Bizim RNN katmanımızdır
    
        #RNN' output katmanı olan fully connected layerı bağlayacağız:
        self.fc = nn.Linear(hidden_size, output_size) #output_size: çıktı boyutu ve 1 olacak çünkü tahmin etmeye çalıştığımız değer 1 adet
    
    
    def forward(self, x): #forward propagation yaptığımız metot
        #forward'da yukarıda oluşturduğumu rnn katmanını ve fully connected katmanı olan outputu birbirine bağlayacağız
        out, _ = self.rnn(x) #x'i input olarak rnn'e verdik ve out olarak çıktıyı aldık. (out: fc layerın inputudur)
        out = self.fc(out[ : , -1, : ]) #tüm satırlar, en sonuncusu, tüm satırları aldık. Son zaman adımındaki çıktıyı aldık ve fc layera bağladık
        return out
        

#şu an denemek için yaptık
model = RNN(1, 16, 1, 1 ) #input_size, hidden_size, output_size, num_layers





# %% 4- RNN'i eğitelim (Training)

""" Modelin hiperparametrelerinin ne olduğunu belirleyelim: """
seq_length = 50   # kullanacağımız input dizisinin boyutu
input_size = 1    # input dizisinin boyutu
hidden_size = 16  # rnn'in gizli katmandaki düğüm sayısı
output_size = 1   # rnn'in çıktısının boyutu. 1 olacak çünkü 1 tane değeri tahmin ediyor
num_layers = 1    # rnn katman sayısı
epochs = 20       # training için gerekli olan epoch sayısını tanımladık. modelin kaç kez tüm veri seti üzerinde eğitileceğidir
batch_size = 32   # her bir eğitim adımında kaç örneğin kullanılacağı
learning_rate = 0.001  # optimizasyon algoritması için öğrenme oranı (hızı)

""" Veriyi Hazırlayalım """
X, y = generate_data(seq_length) #X->input, y->target, seq_length -> verinin ne kadar geçmişe bakacağını belirleyen parametre 
X = torch.tensor(X, dtype = torch.float32).unsqueeze(-1) # Numpy dizisini pytorch tensorüne çevirdik ve boyut ekledik. -1 -> en son boyuta yeni bir eksen ekle demektir. [950, 50, 1] olur
y = torch.tensor(y, dtype= torch.float32).unsqueeze(-1) #-1 -> orayı kendin belirle diyor  [950,1] olur

""" Pytorch veri setini oluşturalım """
dataset = torch.utils.data.TensorDataset(X, y) #Pytorch dataset oluştu
dataLoader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True) # Veri yükleyici oluşturduk. (Loader ile besleriz. dataset'i kendisi batchlere ayıracak bizim yerimize. Ve shuffle ile veriyi karıştırsın dedik)

""" Modeli tanımlayalım   -- Loss ve Optimizer belirledik """
model = RNN(input_size, hidden_size, output_size, num_layers) #input size =1, hidden size = 16, output size =1, num_layers = 1 yaptık
criterion = nn.MSELoss() #Loss fonksiyonumuzdur. (Regresyon problemi çözeceğiz. Regression probleminde optimizer ve loss fonksiyonunu farklı seçmemiz gerekiyor.) Mean Squared Error Loss Fonksiyonunu seçtik (Ortalama kare hatası)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Optimizasyon. (model.parameters()'ı input olarak verdik ve learning rate'i verdik)

""" Modelin eğitim döngüsünü tamamlayalım """
for epoch in range(epochs):
    for batch_x, batch_y in dataLoader: # DataLoader bize image ve image'e karşılık gelen etiketleri return ediyordu. Bizim veri setimiz image değil burada X'ler ve onlara karşılık gelen etiketler de y'dir
        optimizer.zero_grad() #gradyanları sıfırla
        pred_y = model(batch_x) #modele input olarak bacth_x'i verdik o da bize output'u yani pred_y'yi verir
        loss = criterion(pred_y, batch_y ) #input olarak pred_y yani modelin tahmin ettiğini ve batch_y'yi gerçekte olanı alacak ve bize loss değeri return edecek
        #Bu loss değerini kullanarak back propagation (geri yayılım) yapacağız
        loss.backward() #geri yayılım ile gradyanları hesapladı
        optimizer.step() #optimizerı kullanarak ağırlıkları güncelledik
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}") #epoch 1'den başlasın


# %% 5- Test edelim (Testing)

""" 2 adet test verisi oluşturalım """
#np.linspace: belirli bir aralığı eşit parçalara bölerek sayı dizisi oluşturmaya yarayan bir NumPy fonksiyonudur.
X_test = np.linspace(100, 110, seq_length).reshape(1,-1) #100'den başlasın 110'a kadar gitsin ve seq_length kadar olacak
# reshape(1, -1) --> dizinin şeklini (shape) değiştirir. 1: İlk boyutun 1 olmasını sağlar (yani 1 adet örnek/satır). -1: İkinci boyutun, verideki toplam eleman sayısına göre otomatik ayarlanmasını sağlar.

y_test = np.sin(X_test) #X_test içindeki her bir değerin sinüs değerini hesaplayarak target değerlerini oluşturur

#X_test -> ilk test verimiz ,   y_test -> test verimizin gerçek değeri

X_test2 = np.linspace(120, 130, seq_length).reshape(1,-1) #ikinci test verimiz
y_test2 = np.sin(X_test2) #ikinci test verimizin gerçek değeri


""" Numpy'dan Pytorch tensorlerine çevirelim ve boyut ekleyelim """
X_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1) #numpy dizisi olan y_test verisini pytorch tensore dönüştürür. veri tipini float 32 bit olarak sabitledik RNN bu hassasiyette çalışır. (-1) ile de en sona bir boyut daha ekler çünkü RNN 3 boyutlu formatta bekler 
X_test2 = torch.tensor(y_test2, dtype=torch.float32).unsqueeze(-1)


""" Modelimizi kullanarak prediction yapalım """
model.eval() #evaluate moduna aldık
pred1 = model(X_test).detach().numpy() #ilk test verisi için tahmin yapma. 
pred2 = model(X_test2).detach().numpy()

#detach() -> bu Tensor'u hesaplama grafiğinden ayırır. Çünkü artık eğitim yapmıyoruz, sadece sonucu görmek istiyoruz. Bu işlem belleği rahatlatır ve hatayı önler.
#numpy() -> PyTorch Tensor'unu standart bir NumPy dizisine çevirir.


""" Sonuçları görselleştirelim """
plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker = "o", label =" Training dataset") #X ekseni değerleridir -> 0'dan 100'e kadar, elindeki veri sayısı (len(y)) kadar eşit aralıklı nokta oluşturur.  y: Y ekseni değerleridir (eğitim verilerin/sinüs değerlerin)
plt.plot(X_test.numpy().flatten(), marker ="o", label="Test 1") #test verilerini numpy dizisine dönüştürdük ve tek boyutlu hale getirdik
plt.plot(X_test2.numpy().flatten(), marker = "o", label="Test 2")

plt.plot(np.arange(seq_length, seq_length+1), pred1.flatten(), "ro", label="Prediction 1")
plt.plot(np.arange(seq_length, seq_length+1), pred2.flatten(), "ro", label="Prediction 2")
#np.arange(seq_length, seq_length+1) -> X eksenindeki konumu belirler. Eğer elinizde seq_length kadar (örneğin 10 tane) geçmiş veri varsa, modelin tahmin ettiği "yeni" değer 11. sırada olmalıdır. np.arange(10, 11) kodu bize sadece [10] değerini döndürür.
# "ro" -> red ve circle olacak

plt.legend()
plt.show()



