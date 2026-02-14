#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem tanımı: CIFAR10 veri setini kullanarak sınıflandırma problemi çözeceğiz.
CIFAR10 veri seti 60 bin tane küçük renkli görselden oluşur. Bu görseller 10 farklı sınıfa(kategoriye)
eşit olarak dağıtılmıştır.

-> Görsel Boyutu: Her görsel 32x32 pikseldir
-> Renk: Görseller 3 kanallıdır (RGB - Kırmızı, Yeşil, Mavi).
-> Dağılım: 50.000 görsel eğitim (training), 10.000 görsel ise test (testing) için ayrılmıştır.


"""

# %%  1- Import Libraries (Kütüphaneleri içeri aktarma)
import torch #pytorch kütüphanesi
import torch.nn as nn #cnn katmanlarını oluşturmak için sinir ağı katmanlarını kullanacağız
import torch.optim as optim #optimizasyon algoritmasını için
import torchvision #görüntü isleme icin
import torchvision.transforms as transforms #görüntü dönüşümleri yapmak için
import matplotlib.pyplot as plt #görselleştirme
import numpy as np

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # MAC için
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # NVIDIA için (Windows + Linux)
device = "cpu"


#  2- Veri Setini yükleyeceğiz. (Load Dataset)

#batch_size: her iterasyonda işlenecek veri sayısı
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), #görüntüyü tensora çevirir ve scale ederiz
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #mean ve standart sapmaya göre scaling yapar.
        # 3 tane 0.5 yazma nedenimiz renkli görüntü mü yoksa siyah beyaz görüntü mü olup olmamasından kaynaklıdır
        # Yani biz burada RGB kanallarına 0.5'lik normalizasyon uyguluyoruz.
        ])


    # CIFAR10 veri setini indirelim ve eğitim test kümelerini oluşturalım:
    #root: Görüntüleri nereye indireceğimizdir. cnn klasörü içinde data adında klasör oluşturur ve veri setini içine indirir.
    #train : True: Eğitim verisini alır (50 bini eğitim içindir) 
    #download: True -> Eğer veri seti bilgisayarda yoksa internetten indirir varsa tekrar indirmez
    #transform: Görüntülere uygulanacak ön işleme adımları
    train_set = torchvision.datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train = False, download = True, transform = transform)



    #Pytorch Veri yükleyicisini oluşturacağız
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader





#%%  3- Veri setini görselleştireceğiz. (Data Visualization)


#Bir tane görüntüyü görselleştirme işlemi:
def imshow(img): #içine bir tane görsel alacak
    #Biz veri setini indirirken transform'u aktif etmiştik yani görseller normalize edilmişti ama bu görüntüyü görselleştirirsek
    # bozulacaktır çünkü siyah beyaz değiller. Bu yüzden ters normalizasyon uygulayalım,
    # yani verileri normalize etmeden önceki hallerine geri dönüştürelim böylece görselleştirme daha kolay olur
    
    img = img / 2 + 0.5  #ters normalizasyon uyguluyoruz
    np_img = img.numpy() #Tensor formatına çevirdiğimiz görüntülerimizi geri numpy formatına çeviriyoruz
    plt.imshow(np.transpose(np_img, (1, 2, 0))) # 3 kanal için yani RGB renkleri için doğru şekilde sıralama göstererek transpoz ettik
    plt.show()


# Veri kümesinden örnek görselleri alalım
def get_samples_images(train_loader): #train veri setinden görselleştirme yapacağımız için
    data_iter = iter(train_loader) #sıradaki batchi ver diyor
    images, labels = next(data_iter) #ilk batchden görüntü ve etiketleri almamızı sağlar. 64 görüntü, 64 etiket
    return images, labels


#Görselleştirme işlemine geçelim
def visualize(n):
    train_loader, test_loader = get_data_loaders() #şu an sadece görselleştirme yapmak istediğimiz için burda çağırıyoruz yoksa get_data_loadersı ileride başka yerde yazacağız
    
    # n tane veri görselleştirme işlemi yapalım:
    images, labels = get_samples_images(train_loader)
    plt.figure() #yeni ve boş bir grafik alanı oluşturur
    for i in range(n):
        plt.subplot(1, n, i+1 ) #1 satır, n tane sütun, i+1. indeksteki sütunu dolduracak
        imshow(images[i]) #görselleştirir
        plt.title(f"Label: {labels[i].item()}") #labels[i] -> i. görüntünün gerçek etiketi, .item() -> Tensor içindeki sayıyı alır python int'e çevirir
        plt.axis("off") # x ve y eksenini kapattık 
    plt.show()
    
#visualize(10) #bizim için 10 tane görseli görselleştirecek
    
    


#%% 4- CNN Modelinin tanımlanması, inşaa edilmesi (Build CNN)

# İlk olarak class oluşturacağız ve sınıfımızın 2 tane fonksiyonu olacak. İlki constructor olan init fonksiyonu olacak. 
# init fonksiyonu : CNN modelimizi oluşturmak için gerekli olan bileşenleri içerisinde barındırır.
# Diğer fonksiyon ise Forward fonskiyonudur bu da CNN bileşenlerini birbirine bağlayan, forward propagation yaptığımız kısım olacak.
#  #CNN classımız Pythorch'da bulunan NN classından inherit edecek.

class CNN(nn.Module):   #pytorch'un nn.Module sınıfından miras alır
    def __init__(self):
        #CNN sınıfımızı nn modülünden inherit edelim:
        super(CNN, self).__init__() # üst sınıfın (nn.Module) init fonksiyonunu da çalıştırır. Yazılmazsa model düzgün çalışmaz
        
        
        #Şimdi initializer içerisine CNN'de kullanacağım bileşenlerin hepsini tanımlayalım:
        # 1-Convolution Layer'ı tanımlayalım. Tanımlarken 2 boyutlu olarak tanımlayacağız. İlk convolution katmanımızdır:
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        
        # in_channels: inputların yani görüntülerin kanal sayısı. RGB olduğu için 3 diyeceğiz
        # out_channels: filtre sayısı. biz bunu 32 olarak tanımlayacağız
        # kernel_size: convolution layer'da bulunan filtrelerimizin boyutu 3x3'lük matrisler olarak tanımlayacağız
        # padding: dolgudur. convolution sonrası görüntü boyutunun küçülmesini engellemektedir. Yani giriş görüntüsünün her kenarına 1 piksellik 0 ekliyoruz
        
        
        # 2- Aktivasyon fonskiyonu oluşturalım ReLU ile
        self.relu = nn.ReLU()
    
        # 3- Pooling layer tanımlayalım
        # Görüntünün (feature map’in) boyutunu küçültmek ve en güçlü özellikleri korumak için kullanılan bir Max Pooling katmanı tanımlar
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # 2x2 boyutunda görüntüyü parçalara böler, her parçada max değeri seçer ve diğerlerini atar
        #stride = 2 -> pencereyi iki piksel sağa yani ileri kaydırır
        #kernel_size =  2 -> pooling penceresi 2x2, her seferinde 4 pikseli inceler. Çıktı 16x16'dır yani genişlik ve yükseklik yarıya iner
        
        # 4- İkinci Convolution Layer'ı tanımlayalım:
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) # 64 filtreli ikinci convolution katmanı
        # 2. conv layerin girdisi 32 olmak zorunda çünkü conv1'in çıktısı 32'dir.
        #out_channels'a hiperparametre olduğu için kendimiz karar veriyoruz bu yüzden 64 seçtik
        
        
        # 5- Dropout katmanı ekleyelim. Amacı overfitting'i azaltmaktır
          #Dropout, eğitim sırasında bazı nöronları rastgele geçici olarak kapatır 
        self.dropout = nn.Dropout(0.2) # %20 ihtimalle nöron kapatılır yani %20'si devre dışıdır, diğer nöronlarla bağlantısı koparılır
        
        
        # 6- Tam bağlı katmanları (Fully Connected Layers) inşaa edeceğiz:
        self.fc1 = nn.Linear(64*8*8, 128) # 64*8*8-> input size'dır. 64 conv2'den gelen filtre sayısıdır. 8x8 de image boyutudur. 128 ise output size'dır kendimiz belirledik (4096,128)
        
        # Biz bileşenleri belirlerken bunları birleştirirken şunu uygularız: (kanal x görüntü boyutu x görüntü boyutu)
           # 1 # image (3x32x32) -> conv layer (32)lik görüntü elde ederiz-> relu (32) yine boyut değişmedi -> pooling (16) stride=2 old için boyut yarıya düşer  
          # 2  # conv layer (16) -> relu (16)-> pooling (8) yani boyut yarıya indi -> image = 8x8'lik görüntü olur  #forward kısmında bunu yaptıktan sonra elimizde image olacak
        
        #ikinci tam bağlı katman:
        self.fc2 = nn.Linear(128, 10) #output layerımız olduğu için sınıflandırma probleminde kaç sınıf (etiket) varsa output o olur. CIFAR10'da da 10 farklı sınıf var
        
    
    
    
    
    def forward(self, x):
        # Şimdi her bileşeni birbirine bağlayacağız:
            
        """
          image (3x32x32) -> conv layer (32) -> relu (32)  -> pooling (16) boyut yarıya düştü
          conv layer (16) -> relu (16)-> pooling (8) -> image = 8x8'lik görüntü olur  
          flatten işlemi uygulayacağız
          fc1 layer olacak -> ReLu fonksiyonu -> dropout yaparız
          fc2 -> output 
        
        """
        #1- conv1 layerına x girdisini veriyoruz.
        #2- sonrasında onu hemen relu'ya baglıyoruz.
        #3- sonrasında bu reluya bağladığımızı da pooling'e baglıyoruz
        x = self.pool(self.relu(self.conv1(x)))  # """ ilk convolution bloğu """
        
        #4- şimdi ikinci conv2 layerına x'i input olarak verdik
        #5- relu ile bu conv2'yi bağladık
        #6- pooling ile bağladık
        x = self.pool(self.relu(self.conv2(x)))  # """ ikinci convolution bloğu """
    
        # Şimdi Flatten yapacağız. n boyutlu matrisi vektör (tek boyutlu) hale çevirelim. view yöntemini kullanabiliriz bu sefer:
        x = x.view(-1, 64*8*8)  #-1: batch_size'ı sen otomatik hesapla demektir ve kod bu sayede bozulmaz
        # view tensorun şeklini değiştirir, verinin kendisini değiştirmez
        
        # fc1'e x'i input verdik ve fc1 layerı relu ile bağladık ve sonra da dropout ile bağladık
        x = self.dropout(self.relu(self.fc1(x))) #Fully connected layerdır
        
        #output'u gösterelim
        x= self.fc2(x) #Output katmanı da 

        return x
    

#  5- Model oluşturma ve derleme. Loss fonksiyonu ve Optimizer belirlenir.

# CNN classını çağıralım 
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # MAC için
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # NVIDIA için (Windows + Linux)

# Modelimizi tanımlayalım
#device="cpu"
#model = CNN().to(device)



# Loss fonksiyonu ve Optimizer'ı belirleyelim:

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), #Cross Entropy Loss kullanıyoruz çünkü multi class classification problemi çözüyoruz
    optim.SGD(model.parameters(), lr=0.001, momentum = 0.9) #SGD: optimizasyon algoritmasıdır
    
    )

# SGD = Stochastic Gradient Descent, modelin ağırlıklarını hataya bakarak adım adım güncelleyen optimizasyon algoritmasıdır. 
#SGD tüm veri seti yerine küçük bir batch kullanır bu batch her adımda değişir.
#momentum: Geçmiş gradientleri hesaba katar. SGD algoritmasını daha hızlı, stabil yapar. SGD algoritmasının lokal minimumlara takılmasını engeller.




#%%  6- Modeli Eğitelim (Training)

#veri setini yüklemek için train_loader ile loader'ı çağırırız
#criterion: define_loss_and_optimizer'dan gelecek yani loss fonksiyonumuz olacak
# optimizer: define_loss_and_optimizer'dan gelecek optimizerımız olacak 
# training aşamasında kullanacağımız epoch sayısını belirleriz
def train_model(model, train_loader, criterion, optimizer, epochs = 5 ): 
    model.train() #1- Modeli eğitim moduna alalım
    
    train_losses = [] #2- Epoch başına kayıpları(loss değerlerini) saklamak için boş bir liste oluşturalım
    
    for epoch in range(epochs): #3- Belirtilen epoch sayısı kadar eğitim gerçekleştiririz (for döngüsü yani)
        
        total_loss = 0 #4- Toplam kayıp (loss) değerini saklayabilmek için total_loss değeri tanımlarız 
    
        for images, labels in train_loader: #5- Tüm eğitim verileri üzerinde bir iterasyon gerçekleştirebileceğimiz for döngüsü tanımlarız
            images, labels = images.to(device), labels.to(device) #Verileri cihaza taşıdık
        
            optimizer.zero_grad() #6- Gradyanları sıfırlarız. Sıfırlamazsak bir sonraki iterasyona elimizdeki gradyanları iletmek zorunda kalıyorduk bunu da istemiyoruz
            outputs = model(images) #7- Modelden çıktıları alacağız yani Forward Propagation (prediction) yapacağız. Model input olarak images alır. Outputları return etsin bu outputlar labellar (class) oluyor
            loss =  criterion(outputs,labels) #8- Elde ettiğimiz bu y_predicted değerini kullanarak loss fonksiyonu ile birlikte loss değerini hesaplarız. Kriter parametremiz burada input olarak outputları ve gerçek labelları alır
            #Burada criterion'ın amaç şu: Bizim modelimiz ne tahmin etmiş ve gerçekte o görüntü ne olmalıydı. Bu ikisi arasındaki farka bakacak ve loss değerini hesaplayacak
    
            loss.backward() #9- Backward Propagation yaparız ve gradyanları hesaplarız
            optimizer.step() #10- Neural Networkümüz üzerinde bulunan ağırlıkları (parametreleri) güncelleyerek öğrenme işlemini gerçekleştireceğiz.

            total_loss += loss.item() #Loss'u sayıya çevirip toplar. .item() → tensor → Python float
           
        # Bunları yaptıktan sonra; Toplam loss değerini, ortalama kaybı bulacağız bunları listede depolayacağız ve sonra da kayıp grafiğinin çizimini yapacağız.
        
        avg_loss = total_loss / len(train_loader) #Bir epoch boyunca oluşan ortalama kaybı hesapladık. (len(train_loader) -> Bir epochda kaç batch olduğunu söyler)
        train_losses.append(avg_loss) #Ortalama kayıplarımızı listede tuttuk
            
        print(f"Epoch: {epoch + 1} / {epochs}, Loss: {avg_loss:.5f}  ") #ilk epoch 0 olacağı için ama çizerekn biz ilkin 1 olarak yazılmasını istediğimiz için epoch + 1 deriz görselleştirme için. epochs: toplam epochdur.
            
            
    # Kayıp (loss) grafiği: (Normalde train_model fonku içerisinde kayıp grafiği çizdirmeyiz. Bu fonkun asıl amacı CNN'ü eğitmektir çıktılarını normalde başka yerde çizdiririz.)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker="o", linestyle="-", label="Train Loss") #1'den başlasın toplam epoch sayımız + 1 kadar gitsin.
    # Epoch numaraları Örn: epoch = 5 ise -> [1, 2, 3, 4, 5]
    #train_losses -> Her epoch sonunda hesaplanan ortalama loss değerleri
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
    
#train_loader, test_loader = get_data_loaders() #get_data_loaders train_loader ve test_loader return ediyordu
#model = CNN().to(device)
#criterion, optimizer = define_loss_and_optimizer(model) #bu loss fonku ve optimizerı belirlediğimiz fonksiyon içine model'i input alıyor. İçinde bir tane Loss fonku bir tane de Optimizer return ediyor burda da criterion ve optimizer'a denk geliyor
#train_model(model, train_loader, criterion, optimizer, epochs = 10) 

#Epoch sayısını 10 yaptık aslında çok değil az bu yüzden öğrenmeyi çok gerçekleştiremeyecek ama training çok uzun sürüyor ama algoritmayı Gpu üstünde çalıştırırsak daha hızlı çalışır.
#Epoch sayımız 1.05'e kadar düştü biz epoch arttırsaydık büyük ihtimalle biraz daha düşüş gerçekleştirecekti ve yataya bağlayacaktı. Yataya bağlaması öğrenmiş olduğu anlamına gelir. 
#Yataya bağlarsa biz loss değerimizi azaltamıyoruz demektir yani model öğrenebileceği kadar öğrenmiştir.





#%%  7- Modeli Test edelim (Testing)

def test_model(model, test_loader, dataset_type):

    #1- modelimizi değerlendirme moduna alıyoruz
    model.eval()
    
    #2- Toplam tahmin sayacı ve toplam veri sayacı gibi iki tane int değer tutan değişken tanımladık
    correct = 0 #doğru tahmin sayacı
    total_data = 0 #toplam veri sayacı
    
    #3- Gradyan hesaplama gereksiz olduğu için kapatırız ve for döngüsü ile test veri kümemiz üzerinde iteratif bir şekilde tahminleri gerçekleştirelim.
    #4- Yapmış olduğumuz tahminleri (predictions), gerçek etiketlerle karşılaştırıp bir accuracy değeri ortaya çıkaralım
    with torch.no_grad(): #gradyan hesaplaması gereksiz olduğu için kapattık
        for images, labels in test_loader: #Tüm test verileri üzerinde iterasyon gerçekleştireceğiz
            images, labels = images.to(device), labels.to(device) #Verileri cihaza taşıdık
            
            outputs = model(images) #modelimiz tahmin (prediction) gerçekleştiriyor ve model input olarak images alır. Tahmin sonucunda bir tane output return eder
            #Biz bu outputları gerçekte olması gereken etkiletlerle karşılaştıracağız ve accuracy değeri elde edeceğiz
            _, predicted = torch.max(outputs, 1) #Elde ettiğimiz outputlar içerisindeki en yüksek olasılıklı sınıfı seçelim (outputs tensörünün 1. boyutu boyunca (sınıf boyutu) en büyük değeri bulur. 
            #Geriye 2 değer return eder -> _, predicted bu da indexi ve predict ettiğimiz değerdir
            #Genelde testte output şu şekildedir: outputs.shape == (batch_size, num_classes)
            
            total_data += labels.size(0) #Toplam veri sayısıdır. labels.size(0)->0. boyuttaki eleman sayısını verir Yani: batch içindeki örnek sayısı
            correct += (predicted == labels).sum().item() #Modelin kaç tane doğru tahmin yaptığını sayar. predicted (tahmin ettikleri) == labels (gerçek değerler) bu boolean tensor döner True/False diye
            #.sum() -> True False olanı 1-0 şekline dönüştürür ve doğru sayısını toplar mesela 2 True varsa tensor(2) yazar. .item() ise bunu Python int'e çevirir ve direkt 2 yapar.
            
    print(f"{dataset_type} accuracy: {100 * correct / total_data} % ") #Doğruluk oranını yüzde olarak ekrana yazdırdık

#Modelimizi test edecek kod:
# test_model(model, test_loader, dataset_type = "Test") #test veri setimiz # Test accuracy: 61.74 % 
# test_model(model, train_loader, dataset_type = "Training") # Training accuracy: 64.178 % 

#Eğer training accuracy çok yüskek, test accuracy çok düşük çıkarsa overfitting vardır.
#Eğer training accuracy çok düşük, test accuracy çok yüksek çıkarsa underfitting (öğrenememe) vardır.
#Bir çalışma yaptıktan sonra o veri seti ile daha önce gerçekleştirilen Benchmark'lara bakarsanız sizin yaptığınız çalışma sonucunda elde ettiğiniz accuracy değerinin yeterli olup olmadığı ortaya çıkar.


# %% Main program Execute - Ana program çalıştıralım

if __name__ == "__main__":
    #1- Veri seti yüklensin
    train_loader, test_loader = get_data_loaders() #get_data_loaders train_loader ve test_loader return ediyordu
    
    #2- Görselleştirme
    visualize(10) #görselleştirme yapar
    
    #3- Training
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model) #bu loss fonku ve optimizerı belirlediğimiz fonksiyon içine model'i input alıyor. İçinde bir tane Loss fonku bir tane de Optimizer return ediyor burda da criterion ve optimizer'a denk geliyor
    train_model(model, train_loader, criterion, optimizer, epochs = 10) 

    #4- Test
    test_model(model, test_loader, dataset_type = "Test") #test veri setimiz # Test accuracy: 61.74 % 
    test_model(model, train_loader, dataset_type = "Training") # Training accuracy: 64.178 % 

#Biz neden mainde yazıyoruz? Çünkü algoritmamızı geliştirdikten sonra mainde çalıştıracağız sonrasında da bu artık gerçek ürüne çevrilme aşamasında API'ye çevrilmesi gerekiyor ve burdan yola çıkarak ilerliyoruz.

