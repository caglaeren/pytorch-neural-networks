# -*- coding: utf-8 -*-
"""
Problem tanımı: mnist veri setini kullanarak rakam sınıflandırma projesi.
Mnist Veri Seti:
ANN: Artificial Neural Networks (Yapay sinir ağları)

"""
""" # %% yazınca section oluşturmuş olduk."""

# %%library (kütüphaneler yüklenecek)

import torch   #diyerek pytorch kütüphanesini import edebiliriz ve tensor işlemlerini gerçekleştirmek için kullanacağız
import torch.nn as nn  #yapay sinir ağır katmanlarını tanımlamak için kullanırız
import torch.optim as optim #optimization kısaltması, optimizasyon algoritmalarını içeren pytorch modülü
import torchvision #Görüntü işleme ve önceden eğtiilmiş modeller için kullanılır. Mnist datasetini içeri aktarmamızı sağlar.
import torchvision.transforms as transforms # görüntü dönüşümleri yapmak için kullanılır
import matplotlib.pyplot as plt #görselleştirme


# Optional Code: Cihazı belirleme kodu.
#MPS: Macos içindir.
#Cuda: Nvidia (Windows + Linux)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# veri setini yükleyeceğiz - dataloading

#batch: veriyi kaçlı paketler halinde tanımlayacağımız
def get_data_loaders(batch_size=64): #her iterasyonda işlenecek veri miktarı (64'lü paketler halinde)
    
    #görüntüyü tensorlara dönüştüreceğiz ve pixel değerlerini -1 ile 1 arasında ölçeklendireceğiz
    #compose içinde liste oluşturduk
    #listenin içinde transforms.ToTensor() dönüşümü içinde python ile elde edilen görüntü veya klasik bir numpy array olabilir
    #biz bunu alıp tensor formatına çevireceğiz, scaling yapacağız.
    
    transform = transforms.Compose([
        transforms.ToTensor(), #dönüşümü içinde görüntüyü veya klasik bir numpy array olabilir bunu tensora çevireceğiz scale edeceğiz. 0-1 arasında ölceklenir
        transforms.Normalize((0.5,), (0.5,))  #mean ve standart sapmaya göre scaling yapar. piksel değerlerini -1 ile 1 arasında ölçekler
    ])
    
    
    #Mnist veri setini indireceğiz ve eğitim-test kümelerini oluşturacağız.
    #root: görüntüleri nereye indireceğimizdir. "./data" ise 1_ann içinde data klasörüne bunları oluştur demektir. data klasörü varsa içine yazar direkt.
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform = transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform = transform)
    
    
    #Pytorch veri yükleyicilerisini oluşturalım:
    #veriyi 64'lü paketler (batch) halinde yükler
    #shuffle: veriyi karıştırsın yani veri sıralı olmasın
    #kontrollü test yapabilmek için onu karıştırmayız.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    return train_loader, test_loader
    
#train_loader, test_loader  = get_data_loaders() #en sonda bunları çağırdık sonradan

    

# veri setini basit şekilde görselleştireceğiz - data visualization

#loader -> train_loader veya test_loader'ı kullanabilliriz bu yüzden loader olarak aldık
def visualize_samples(loader, n):
    images, labels = next(iter(loader)) #ilk batchden görüntü ve etiketleri alalım. 64 görüntü ve 64 etiket.
    #print(images[0].shape) #ilk verinin özelliklerine, boyutuna baktık. (1, 28, 28) verdi.
    fig, axes = plt.subplots(1, n, figsize=(10, 5)) #nrows, ncols ve n farklı görüntü için görselleştirme alanı hazırladık
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray") #görseli gri tonlamalı göster
        #squeeze(): boyutu 1 olan gereksiz eksenleri siler.
        #imshow: 2D ister ve matrisi (array/tensor) resim olarak ekrana çizdirir
        axes[i].set_title(f"label: {labels[i].item()}") #görüntüye ait sınıf etiketini başlık olarak yazar
        axes[i].axis("off") #x ve y eksenlerini görmek istemiyoruz çünkü görüntü görselleştireceğiz eksenlere gerek yok
    plt.show()
        

#visualize_samples(train_loader, 4) #4 tanesini görselleştirir




# %%Ann modelimizin tanımlanması - define ann model

#Neural network classı oluşturacağız ve bu class, pytorch neural network classının modüllerinden kalıtım alacak
class NeuralNetwork(nn.Module): #pytorch'un nn.module sınıfından miras alıyor
    #classımızın (neural network'ümüzün) bir tane initializer yani constructor'ı olacak
    #ve bir tane de forward propagation yapabilmemiz için gerekli olan forward fonksiyonu olacak.
    
    #NN inşa etmek için gerekli olan bileşenleri tanımlayalım:
    def __init__(self):
        #inherit edebilmesi için
        super(NeuralNetwork, self).__init__() #Böylece pytorch'da bulunan nn.Module'ü içerisindeki layerları kullanabilir hale geldik
        
        #Şimdi ANN'ümüzü inşaa edeceğiz. 
        # İlk olarak 2 boyutlu görüntüleri tek boyutlu hale çevirmeliyiz yani vektör haline.
        self.flatten = nn.Flatten() #çok boyutlu veriyi tek boyuta çevirir
        
        
        # Şimdi ilk tam bağlı katmanımızı oluşturalım. (fc:fully connected)
        #nn.Linear(input,output) alıyor
        self.fc1 = nn.Linear(28*28, 128) #28*28=784 input'un size'ı, 128 = output size ve output'u biz belirledik
        
        # Aktivasyon fonksiyonu oluştur. (Tam bağlı katmanını olutşurduktan sonra aktivasyon fonku ile beslemeliyiz.)
        #ReLu: Sinir ağlarında kullanılan aktivasyon fonskiyonudur.
        self.relu = nn.ReLU()
        
        # İkinci tam bağlı katmanı oluşturalım.
        self.fc2 = nn.Linear(128, 64) #genellikle output bu şekilde azalarak gider, yarıya düşer.
        #128 -> input size
        #64 -> output size
        
        
        #Çıktı katmanını oluştur. (fc3 yerine output da yazılabilir.)
        self.fc3 = nn.Linear(64, 10) #64 -> input size, 10 -> output size
        #output'a 10 yazmak zorundayız çünkü Mnist dataseti 10 tane sınıftan oluşuyor. (0-9)
        
        
    #Şimdi her bir bileşeni birbirine bağlayacağız
    def forward(self, x):  # forward propagation : ileri yayılımdır. giriş olarak x alsın (x = görüntü)
        # initial x = 28*28  -> x yani görüntümüz 28*28'lik görüntüden oluşur.
        # Biz bunu flatten ile tek boyuta çevirip düzleştireceğiz. Yani 784'lük bir vektör haline gelir.
        x = self.flatten(x)
        
        #x'imizi ilk tam bağlı katmana input olarak verelim
        x = self.fc1(x) # birinci tam baglı katman
        x = self.relu(x) # aktivasyon fonku
        x = self.fc2(x) # ikinci tam baglı katman
        x = self.relu(x) # aktivasyon fonku tanımladık. Bunu yukarıda tanımlamamıştık
        x = self.fc3(x) # output katmanı
        
        return x  # bize x diye çıktı üretir yani modelimizin çıktısını return ediyoruz




# modelimizi oluşturacağız ve derleyeceğiz - create model and compile
device="cpu"
# model = NeuralNetwork().to(device) #ben cpu'da çalıştıracağım için böyle yazdım. (Aşağıda kodu çağırdım)

#Loss (kayıp) fonksiyonu ve optimizasyon algoritmasını belirleyelim:
# kayıp fonku olarak Cross Entropy Loss kullanacağız. Çok sınıflı sınıflandırma problemlerinde kullanılır. Çapraz entropi hesabı yapar.
# optimizer olarak da adaptive momentum kullanacağız (ağırlıkları güncellemek için)

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # çok sınıflı sınıflandırma için kullanılan kayıp fonksiyonu
    optim.Adam(model.parameters(), lr=0.001) # ağırlıkları adaptive momentum algoritması ile günceller
    #lr:learning rate yazmamıza esasen gerek yok. çünkü adam'ın lr default'u 0.001'dir.
    
    )
# criterion, optimizer  = define_loss_and_optimizer(model) #kriteri ve optimizerı return eder. (Bunu da aşağıda çağırdık)



# %% modeli eğiteceğiz - training

#criterion (kriter): loss fonksiyonumuzdur
#epochs: bu işlemi kaç kere yapacak
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    model.train() # 1- Modelimizi eğitim moduna alacağız.
    
    train_losses = []  # 2- Her bir epoch sonunda ortaya çıkan kayıpları saklamak için bir train loss listesi tanımlarız.
    
    # 3- Belirtilen epoch sayısı kadar eğitim yapacağız.
    for epoch in range(epochs):
        total_loss = 0 #toplam kayıp değeri
        
        # 4- Tüm eğitim verileri üzerinde iterasyon gerçekleştireceğiz.
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) #verileri cihaza taşımış olacak. gpu kullanırsak önemli
            
            #Önemli ve yapmamız gereken şey: Neural newtork'ün nasıl training edileceğidir.
            # 5- İlk olarak elimizde bulunan gradyanları sıfırlamalıyız. (Türevlere dayalı hata yayılımı değerleri)
            # Eğer her iterasyonda sıfırlamazsak, bir önceki iterasyondan gelen değerler de kullanılır.
            optimizer.zero_grad() #gradyanları sıfırladık
          
            
            # 6- Modeli uygula, yani forward propagation oluyor.
            #modelimizin imputu images'dır. bu bana prediction'ı yani outputları return eder.
            predictions = model(images)
            
            
            # 7- Loss hesaplayacağız. (y_prediction ile y_real arasında hesaplanır ve aralarındaki hata azsa yani Loss düşükse NN iyi tahmin etmiş demektir)
            loss = criterion(predictions, labels)  #predictionlarımız ile labellar arasındaki doğru hesaplayıp hesaplamadığımızı söyler
            
            
            # 8- Eğer hata çoksa öğrenme işlemi (back forward-geri yayılım) yani gradyan hesaplama gerçekleştirilir.
            loss.backward() #geri yayılımla gradyan hesapladık
        
        
            # 9- Ağırlıklar (weights) güncellenir.
            optimizer.step() #step burda update anlamına gelir

            total_loss = total_loss + loss.item() #loss aslında sayı olarak return etmeyecek liste olarak return edecek onun içindeki sayıyı item ile alabiliriz
            
        avg_loss = total_loss / len(train_loader) #ortalama kayıp hesaplar
        train_losses.append(avg_loss) #Hesaplanan ortalama kayıp listeye eklenir
        #Epoch ve Kayıp bilgilerini ekrana yazdıralım:
        print(f"Epoch {epoch+1}/{epochs},  Loss: {avg_loss : .3f}") #saymaya 0'dan başladığı için epoch+1 deriz. .3f ise virgülden sonra 3 basamak göstrir.

    #Kayıp grafiğini çizerek görselleştirelim: (Loss graph)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle="-", label="Train Loss" ) #1'den başla, epochs+1'e kadar git (epochs + 1 yazmazsak 0'dan başlar ama yazınca 1. epochdan başlar)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend() #label (etiket) gözüksün diye
    plt.show()
    
# train_model(model, train_loader, criterion, optimizer, epochs=3) #Bu kodu da aşağıda çağırdık




# %% modeli test edeceğiz - testing ve görselleştireceğiz

def test_model(model, test_loader):
    model.eval() #modeli evaluation (değerlendirme) moduna alıyoruz
    correct = 0  # modelin yaptığı doğru tahminleri tutmak için tanımlanan sayaç
    total = 0  # toplam veri sayısını tutmak için sayaç
    
    with torch.no_grad(): #Gradyan hesaplama gereksiz olduğunda kapatır
        for images, labels in test_loader: #test veri kümesini döngüye aldık
            images, labels = images.to(device), labels.to(device) #verileri cihaza taşı
            predictions = model(images) #modele images'ları verirsek tahminleri return eder
            
            #Biz en yüksek olasılığa sahip sınıfı seçeceğiz
            #torch.max(tensor, dim) -> verilen boyutta en büyük değeri bulur, max değeri ve o değerin indexini döndürür. dim sınıfın boyutu
            # _ ise burada 'buna ihtiyacım yok' dmeektir. yani bizim kodumuzda max değerini atıyoruz. indexleri (class id) alıyoruz o da koddaki predicted olan.
            #predicted -> index
            _, predicted = torch.max(predictions, 1) #her örnek için en yüksek skora sahip sınıfın indeksini (etiketini) seçer
            total += labels.size(0) #toplam veri sayısını günceller. (Her batch’te o batch’teki örnek sayısını toplam veri sayısına ekle)
                #labels.size(0) -> tensorun 0. boyutunun uzunluğudur
                
            correct += (predicted == labels).sum().item() #tahmin edilen değerler ile etiketler (gerçek değerler) birbirine eşitse True döner. 
            # .sum() pythonda True ->1,, False ->0 dır. sum True'ları sayar. Sonuç tek elemanlı pytorch tensor olur. tensor(18) gibi yani true'ların toplamını verir.
            # .item() ise -> bu tek elemanlı tensor'u normal python sayısına çevirir -> 18 şeklinde (int olur)  (correct normal bir sayıdır, tensor değildir.)
            
    print(f"Test Accuracy: {100 * correct / total:.3f}%")
    
# test_model(model, test_loader) #Bu kodu da aşağıda çağırdık
            
            
# %% Main (ana program)

#Şimdiye kadar tanımladığımız bu 6 fonksiyonu aşağıdaki gibi birleştirebiliriz
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders() #veri yükleyicilerini alalım
    visualize_samples(train_loader, 5) #train veri setinden 5 tanesini görselleştirsin
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model) #içine modeli input alır, kriter ve optimizer döner
    train_model(model, train_loader, criterion, optimizer) #epochs = 10 default alır. modeli train ettik
    test_model(model, test_loader) #modeli test ediyoruz
    
            
            
            
            
            
            
            
            
            
