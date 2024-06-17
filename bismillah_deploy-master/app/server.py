from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io
import os

app = FastAPI()

# with open('app/model.json', 'r') as json_file:
#     json_saved_model = json_file.read()

# model = tfjs.converters.load_keras_model(json_saved_model)
# model.load_weights('app/bin/')

model = load_model('app/explorentt_mobilenet_v2.h5')

class_names = np.array([
    'Air Terjun Oenesu',
    'Air Terjun Tanggedu',
    'Air Terjun Tesbatan',
    'Bukit Wairinding',
    'Danau Kelimut',
    'Gua Rangko',
    'Lawa Darat Gili',
    'Lingko Spider Web Rice Field',
    'Pandar Island',
    'Pantai Koka',
    'Pantai Kolbano',
    'Pantai Mandorak',
    'Pantai Oetune ',
    'Pantai Waecicu',
    'Pantai Walakiri',
    'Pantai Watu Parunu',
    'Pink Beach',
    'Pulau Kalong',
    'Pulau Kanawa',
    'Rumah Budaya Sumba',
    'Savana Puru Kambera',
    'Taman Nostalgia Kupang',
    'Wae Rebo Village',
    'Waikuri Lagoon',
    'Wisata Adat Kampung Todo'])  

@app.get('/')
def read_root():
    return {'message': 'Image classification'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an uploaded image.

    Args:
        file (UploadFile): An image file to predict.

    Returns:
        dict: A dictionary containing the predicted class.
    """
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    response = {'predicted_class': predicted_class}

    if predicted_class == 'Air Terjun Oenesu':
        response.update({
            'image' : 'https://www.google.com/imgres?q=air%20terjun%20oenesu&imgurl=https%3A%2F%2Fpariwisatakabkupang.org%2Fwp-content%2Fuploads%2F2023%2F12%2F20220215_111011-011-1-scaled.jpeg&imgrefurl=https%3A%2F%2Fpariwisatakabkupang.org%2F2019%2F03%2F01%2Fair-terjun-oenesu-3%2F&docid=eVKOguAYLMQ6DM&tbnid=Iu4iYQnA6-rjpM&vet=12ahUKEwiurcO95uGGAxUD8DgGHUvYDsAQM3oECFEQAA..i&w=2560&h=1920&hcb=2&ved=2ahUKEwiurcO95uGGAxUD8DgGHUvYDsAQM3oECFEQAA',
            'artikel': ("Air Terjun Oenesu adalah salah satu tempat wisata yang ada di Kabupaten Kupang, "
                        "Nusa Tenggara Timur yang terkenal akan keindahan alamnya. Air terjun ini adalah "
                        "air terjun bertingkat dengan aliran air yang jernih dan kolam natural yang disokong "
                        "oleh hutan hijau yang rimbun. Akses jalan menuju air terjun ini sangat mudah. Harga "
                        "tiket yang ditawarkan hanya 5.000 per orang. Oenesu sangat digemari oleh para pengendara "
                        "jalan untuk berenang dan bersantai karena suasana yang sangat tenang dan asri."),
            'lokasi': 'https://maps.app.goo.gl/hvtjpgSFLFtAQCAF7',
            'rating': 4.5
        })

    elif predicted_class == 'Air Terjun Tanggedu':
        response.update({
            'image' : 'https://www.google.com/imgres?q=air%20terjun%20tanggedu&imgurl=https%3A%2F%2Fauthentic-indonesia.com%2Fwp-content%2Fuploads%2F2019%2F04%2Ftanggedu-waterfall.jpg&imgrefurl=https%3A%2F%2Fraksantara.com%2Fair-terjun-tanggedu%2F&docid=AiV7ewpY83WBcM&tbnid=wuQMHvmfJR4TeM&vet=12ahUKEwivgc_T5uGGAxX7UGwGHcdLAUYQM3oECDYQAA..i&w=1075&h=789&hcb=2&ved=2ahUKEwivgc_T5uGGAxX7UGwGHcdLAUYQM3oECDYQAA',
            'artikel': ("Air Terjun Tanggedu di Pulau Sumba, Nusa Tenggara Timur, Indonesia, menawarkan pemandangan alam "
                        "spektakuler dengan airnya yang jatuh dari tebing curam setinggi sekitar 100 meter. Terletak di "
                        "Kabupaten Waikabubak, air terjun ini dikelilingi oleh hutan tropis yang hijau. Perjalanan menuju "
                        "air terjun melibatkan pendakian dan trekking yang menantang namun sebanding dengan pemandangan "
                        "alam yang luar biasa. Bagi penggemar petualangan alam, Air Terjun Tanggedu adalah destinasi yang "
                        "sempurna untuk mengunjungi di Pulau Sumba."),
            'lokasi': 'https://maps.app.goo.gl/tukyxZEE1rTBHnLm9',
            'rating': 4.8
        })

    elif predicted_class == 'Air Terjun Tesbatan':
        response.update({
            'image' : 'https://www.google.com/imgres?q=air%20terjun%20tesbatan&imgurl=https%3A%2F%2Fpariwisatakabkupang.org%2Fwp-content%2Fuploads%2F2023%2F12%2Fair-terjun-tesbatan-1.jpg&imgrefurl=https%3A%2F%2Fpariwisatakabkupang.org%2F2019%2F03%2F01%2Fair-terjun-tesbatan%2F&docid=YAWrTO6dNjQS4M&tbnid=26qYZ9HW6Wd6fM&vet=12ahUKEwjP2oDk5uGGAxVs-zgGHfe1BNwQM3oECBgQAA..i&w=800&h=420&hcb=2&ved=2ahUKEwjP2oDk5uGGAxVs-zgGHfe1BNwQM3oECBgQAA',
            'artikel': ("Air Terjun Tesbatan menawarkan pengalaman unik dengan tiga tingkatan yang berbeda. "
                        "Kolam di bawah air terjun memungkinkan pengunjung bermain air sambil menikmati keindahan "
                        "alam. Fasilitasnya meliputi lopo-lopo, area parkir, dan ruang ganti pakaian. Buka setiap "
                        "hari dari pukul 08.00 hingga 18.00, dengan harga karcis masuk yang terjangkau. Dengan segala "
                        "fasilitasnya, Tesbatan adalah tempat yang sempurna untuk bersantai dan menikmati alam."),
            'lokasi': 'https://maps.app.goo.gl/PnKwLQjRYZ4RVu9r9',
            'rating': 4.2
    })

    elif predicted_class == 'Bukit Wairinding':
        response.update({
            'image' : 'https://www.google.com/imgres?q=bukit%20wairinding&imgurl=https%3A%2F%2Fblogger.googleusercontent.com%2Fimg%2Fb%2FR29vZ2xl%2FAVvXsEhw66mb3VO4_xDkyje9-eUMCPWQheSpDEPo1ytV10S3T19FZuu6UNmNKOjGEXyz0ZWbW0Q8ZPj-kJOvdWqVHBx-f-DkJ1viOpqjJADMTmYrfT0s7EpASqhgRFw1EloUaRrkcHqJBd87BXtFIUAjVkl46KsCLbKL7yku_9gsFgW76Bg6D-IPY0L0Xg%2Fs800%2F20230312_234348_0001-01.jpeg&imgrefurl=https%3A%2F%2Fwww.adventurose.com%2F2023%2F03%2Fbukit-wairinding-sumba.html&docid=DWmT3EnbY8e1qM&tbnid=HrNDShcWfZA2fM&vet=12ahUKEwi_sfTz5uGGAxWr1zgGHeVTLsgQM3oECEgQAA..i&w=800&h=533&hcb=2&ved=2ahUKEwi_sfTz5uGGAxWr1zgGHeVTLsgQM3oECEgQAA',
            'artikel': ("Bukit Wairinding terletak di Desa Pambota Jara, Kecamatan Pandawai Sumba Timur, Nusa Tenggara Timur (NTT). "
                        "Daya tarik utama dari Bukit Wairinding tak lain adalah lanskap padang savana yang memanjakan mata. Sejauh mata "
                        "memandang, lekuk-lekuk bukit yang eksotis dan masih asri menjadikan lokasi ini begitu istimewa. Terlebih jika "
                        "waktu mendekati senja, garis cakrawala akan memunculkan warna yang memesona. Selain itu, udara juga tidak terlalu "
                        "panas sehingga wisatawan bisa lebih leluasa berjalan-jalan dan berfoto. Tak jarang wisatawan akan bertemu dengan "
                        "anak-anak Sumba yang tengah menggembalakan hewan ternaknya. Anak-anak Sumba ini juga kerap membantu menunjukkan jalan "
                        "bagi wisatawan. Tak jarang wisatawan mengajak mereka berfoto bersama, atau berfoto dengan kuda yang mereka bawa. "
                        "Suasana ini tentunya tidak bisa didapatkan di kota besar, sehingga akan menjadi pengalaman menarik dan berkesan yang "
                        "bisa diingat ketika pulang."),
            'lokasi': 'https://maps.app.goo.gl/8HYdtcx2wvz85M8XA',
            'rating': 4.8
    })

    elif predicted_class == 'Danau Kelimutu':
        response.update({
            'image' : 'https://www.google.com/imgres?q=danau%20kelimutu&imgurl=https%3A%2F%2Fstatic.promediateknologi.id%2Fcrop%2F0x0%3A0x0%2F750x500%2Fwebp%2Fphoto%2Fp1%2F420%2F2024%2F06%2F14%2FDanau-Kelimutu-Nusa-Tenggara-Timur-85878557.jpg&imgrefurl=https%3A%2F%2Fwww.okocenews.com%2Fhari-ini%2F42012908728%2Fpesona-tiga-warna-danau-kelimutu-keajaiban-alam-di-nusa-tenggara-timur&docid=f0N4S_Bf4xkVTM&tbnid=GL54Djor1DqPoM&vet=12ahUKEwis3IOP5-GGAxWCcGwGHWp8CMcQM3oFCIEBEAA..i&w=750&h=500&hcb=2&itg=1&ved=2ahUKEwis3IOP5-GGAxWCcGwGHWp8CMcQM3oFCIEBEAA',
            'artikel': ("Gunung Kelimutu adalah gunung berapi yang terletak di Pulau Flores, Provinsi Nusa Tenggara Timur. "
                        "Lokasi gunung ini tepatnya di Desa Pemo, Kecamatan Kelimutu, Kabupaten Ende. Gunung ini memiliki "
                        "tiga buah danau kawah di puncaknya. Danau ini dikenal dengan nama Danau Tiga Warna karena memiliki "
                        "tiga warna yang berbeda, yaitu merah, biru, dan putih. Walaupun begitu, warna-warna tersebut selalu "
                        "berubah-ubah seiring dengan perjalanan waktu. Kelimutu merupakan gabungan kata dari 'keli' yang berarti "
                        "gunung dan kata 'mutu' yang berarti mendidih. Menurut kepercayaan penduduk setempat, warna-warna pada "
                        "danau Kelimutu memiliki arti masing-masing dan memiliki kekuatan alam yang sangat dahsyat. Luas ketiga "
                        "danau itu sekitar 1.051.000 meter persegi dengan volume air 1.292 juta meter kubik. Batas antar danau adalah "
                        "dinding batu sempit yang mudah longsor. Dinding ini sangat terjal dengan sudut kemiringan 70 derajat. "
                        "Ketinggian dinding danau berkisar antara 50 sampai 150 meter."),
            'lokasi': 'https://maps.app.goo.gl/BLUBGdrdRpC5oR2r5',
            'rating': 4.7
        })

    elif predicted_class == 'Gua Rangko':
        response.update({
            'image' : 'https://www.google.com/imgres?q=gua%20rangko&imgurl=https%3A%2F%2Fdynamic-media-cdn.tripadvisor.com%2Fmedia%2Fphoto-o%2F0d%2Ff4%2F57%2F2e%2Fgua-rangko-dengan-laguna.jpg%3Fw%3D1200%26h%3D-1%26s%3D1&imgrefurl=https%3A%2F%2Fwww.tripadvisor.co.id%2FAttraction_Review-g1777483-d10846065-Reviews-Rangko_Cave-Labuan_Bajo_Flores_East_Nusa_Tenggara.html&docid=f-kWVWQdWS5VjM&tbnid=0z0aa3lVIGSRZM&vet=12ahUKEwiG5t2g5-GGAxWS2DgGHX3yAm8QM3oECBgQAA..i&w=673&h=373&hcb=2&ved=2ahUKEwiG5t2g5-GGAxWS2DgGHX3yAm8QM3oECBgQAA',
            'artikel': ("Gua Rangko adalah tempat wisata yang terletak di Pulau Gusung, Desa Rangko, Kecamatan Boleng, Kabupaten Manggarai "
                        "Barat, Nusa Tenggara Timur (NTT). Tempat ini menawarkan keindahan gua alami dengan stalaktit khas gua. Di dalamnya, "
                        "pengunjung akan melihat kolam alam yang sangat jernih sehingga batuan stalakmit di dasar kolam terlihat jelas. Jika "
                        "terkena sinar matahari, kolam dengan kedalaman empat meter itu akan terlihat semakin indah. Pengunjung juga dapat "
                        "berenang di kolam ini untuk menikmati kesegaran air alam, namun mereka diminta untuk berhati-hati agar anggota badan "
                        "tidak terkena stalakmit yang terdapat di sekitar kolam."),
            'lokasi': 'https://maps.app.goo.gl/zKNTzGf3r9wgNf5f9',
            'rating': 4.5
        })

    elif predicted_class == 'Lawa Darat Gili':
        response.update({
            'image' : 'https://www.google.com/imgres?q=lawa%20darat%20gili&imgurl=https%3A%2F%2Fdiveconcepts.com%2Fimg%2Fmedia_library%2Fgreat-point-of-view-in-gili-lawa-darat_07072021055712f2679d19184d071c086f62115e76a88a.jpg&imgrefurl=https%3A%2F%2Fdiveconcepts.com%2Fgili-lawa-darat-komodo&docid=qz7GfhpkdfcD0M&tbnid=WBlVd29Q0YeyuM&vet=12ahUKEwjZ-9et5-GGAxW4xzgGHQ9pCwsQM3oECEsQAA..i&w=480&h=480&hcb=2&ved=2ahUKEwjZ-9et5-GGAxW4xzgGHQ9pCwsQM3oECEsQAA',
            'artikel': ("Gili Lawa Darat adalah salah satu pulau kecil yang terletak di Taman Nasional Komodo, di Provinsi Nusa Tenggara "
                        "Timur, Indonesia. Pulau ini terkenal karena pemandangan alamnya yang menakjubkan, termasuk bukit-bukit kecil, "
                        "hamparan padang rumput, dan tebing-tebing batu yang menjulang di sepanjang pantainya. Gili Lawa Darat juga "
                        "merupakan tempat yang populer untuk aktivitas snorkeling dan diving karena terumbu karangnya yang indah dan "
                        "keanekaragaman hayati bawah laut yang kaya. Pemandangan matahari terbenam dari pulau ini juga sangat terkenal, "
                        "memberikan pengalaman yang luar biasa bagi para pengunjung. Keindahan alamnya yang menakjubkan menjadikan Gili "
                        "Lawa Darat sebagai salah satu tujuan wisata yang sangat dicari di wilayah tersebut."),
            'lokasi': 'https://maps.app.goo.gl/zEZCTyYzyE1eEPWq6',
            'rating': 4.9
        })

    elif predicted_class == 'Lingko Spider Web Rice Field':
        response.update({
            'image' : 'https://www.google.com/imgres?q=lingko%20spider%20web%20rice%20fields&imgurl=https%3A%2F%2Fdynamic-media-cdn.tripadvisor.com%2Fmedia%2Fphoto-o%2F03%2F58%2Ff7%2Fae%2Fsawah-sarang-laba-laba.jpg%3Fw%3D1200%26h%3D-1%26s%3D1&imgrefurl=https%3A%2F%2Fwww.tripadvisor.co.id%2FAttraction_Review-g677681-d3675759-Reviews-Lingko_Spider_Web_Rice_Fields_Walking_Tours-Ruteng_Flores_East_Nusa_Tenggara.html&docid=Cgj6Q8RRg5O2iM&tbnid=rmwkHMUVm5_V-M&vet=12ahUKEwiV2v265-GGAxW-SGwGHcG2BwkQM3oECGQQAA..i&w=1200&h=800&hcb=2&ved=2ahUKEwiV2v265-GGAxW-SGwGHcG2BwkQM3oECGQQAA',
            'artikel': ("Lingko Spider Web Rice Field adalah istilah yang mengacu pada pola ladang padi tradisional yang terdapat di "
                        "Sumba, Nusa Tenggara Timur, Indonesia. Ladang-ladang ini terkenal karena polanya yang menyerupai jaring "
                        "laba-laba atau spider web ketika dilihat dari udara. Mereka adalah contoh unik dari sistem penanaman "
                        "terasering tradisional yang masih dipraktikkan oleh masyarakat Sumba. Pola ini tidak hanya memiliki nilai "
                        "fungsional dalam pengelolaan air dan tanah, tetapi juga memiliki makna simbolis dan budaya yang dalam bagi "
                        "masyarakat lokal. Lingko Spider Web Rice Field adalah contoh menakjubkan dari keterampilan tradisional dalam "
                        "pertanian yang telah bertahan selama berabad-abad di Pulau Sumba."),
            'lokasi': 'https://maps.app.goo.gl/dpkiwxWwtcEJJnRH9',
            'rating': 4.3
    })

    elif predicted_class == 'Padar Island':
        response.update({
            'image' : 'https://www.google.com/imgres?q=padar%20island&imgurl=https%3A%2F%2Fwww.indonesia.travel%2Fcontent%2Fdam%2Findtravelrevamp%2Fen%2Fdestinations%2Fbali-nusa-tenggara%2Fthe-wonder-of-padar-island-an-undeniable-travel-wish-list%2Ffajruddin-mudzakkir.jpg&imgrefurl=https%3A%2F%2Fwww.indonesia.travel%2Fgb%2Fen%2Fdestinations%2Fbali-nusa-tenggara%2Flabuan-bajo%2Fthe-wonder-of-padar-island-an-undeniable-travel-wish-list.html&docid=fdkQudi86l5egM&tbnid=q3CwdhRG2rTU9M&vet=12ahUKEwiQ0pHU5-GGAxUzSWwGHciLAtwQM3oFCIUBEAA..i&w=1200&h=800&hcb=2&ved=2ahUKEwiQ0pHU5-GGAxUzSWwGHciLAtwQM3oFCIUBEAA',
            'artikel': ("Padar Island adalah salah satu pulau yang terletak di Taman Nasional Komodo, Nusa Tenggara Timur, Indonesia. "
                        "Pulau ini terkenal dengan pemandangan alamnya yang spektakuler, terutama panorama dari puncak bukitnya yang "
                        "menawarkan pemandangan pantai dengan air berwarna biru dan pasir putih."),
            'lokasi': 'https://maps.app.goo.gl/p9H8cM37AEEW1B4m9',
            'rating': 4.8
    })

    elif predicted_class == 'Pantai Koka':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20koka&imgurl=https%3A%2F%2Fdynamic-media-cdn.tripadvisor.com%2Fmedia%2Fphoto-o%2F17%2F67%2F4a%2F72%2Fpantai-koka.jpg%3Fw%3D1200%26h%3D-1%26s%3D1&imgrefurl=https%3A%2F%2Fwww.tripadvisor.co.id%2FAttraction_Review-g676343-d10162124-Reviews-Koka_Beach-Maumere_Flores_East_Nusa_Tenggara.html&docid=gwum_ikaeSt0xM&tbnid=GGaosWPXwVMTUM&vet=12ahUKEwj15-Ll5-GGAxWAXGwGHUaLGpkQM3oECBkQAA..i&w=717&h=714&hcb=2&ved=2ahUKEwj15-Ll5-GGAxWAXGwGHUaLGpkQM3oECBkQAA',
            'artikel': ("Pantai Koka, juga dikenal sebagai Paga Beach, adalah salah satu tempat paling indah di Flores. Pantai ini memiliki "
                        "pemandangan yang memukau dengan pasir putih yang dikelilingi oleh tebing karang yang ditumbuhi flora tropis. Pantai "
                        "ini adalah pilihan sempurna untuk liburan pantai klasik, menawarkan suasana yang sempurna untuk piknik, pertemuan "
                        "romantis, dan jalan-jalan santai di malam hari di sepanjang pantai. Hembusan angin laut yang segar memberikan sensasi "
                        "yang menyenangkan sepanjang hari, sementara air biru yang jernih, dengan pendekatan yang lembut dan dasar laut "
                        "berpasir, menjamin kondisi yang ideal untuk berenang."),
            'lokasi': 'https://maps.app.goo.gl/P7XJdCq6bHj7twwx9',
            'rating': 4.4
    })

    elif predicted_class == 'Pantai Kolbano':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20kolbano&imgurl=https%3A%2F%2Fstatic.promediateknologi.id%2Fcrop%2F0x0%3A0x0%2F750x500%2Fwebp%2Fphoto%2Fp1%2F583%2F2024%2F04%2F24%2FPicsart_24-04-24_13-08-07-176-2989034274.jpg&imgrefurl=https%3A%2F%2Fwww.beritawisata.com%2Fwisata%2F58312498925%2Fpantai-kolbano-kupang-ntt-pantai-yang-memiliki-batu-kerikil-halus-seperti-pasir&docid=6RFLhRUgJsOQdM&tbnid=JlAtg4k8So7JYM&vet=12ahUKEwjq9-zz5-GGAxVzT2wGHY_zBdkQM3oECDoQAA..i&w=750&h=500&hcb=2&ved=2ahUKEwjq9-zz5-GGAxVzT2wGHY_zBdkQM3oECDoQAA',
            'artikel': ("Pantai Kolbano adalah sebuah destinasi wisata yang terletak di Desa Kolbano, Kecamatan Kolbano, Kabupaten Timor "
                        "Tengah Selatan, Nusa Tenggara Timur. Pantai ini dikenal dengan keunikannya, yaitu tidak memiliki hamparan pasir "
                        "seperti pantai pada umumnya, melainkan batuan. Pantai ini menghadap ke Samudera Hindia dan memiliki bibir pantai "
                        "dengan hamparan batu licin berwarna-warni. Salah satu ciri khas dari pantai ini adalah adanya batu raksasa yang "
                        "disebut ‘Fatu Un’, yang jika dilihat dari samping, batu ini mirip kepala singa."),
            'lokasi': 'https://maps.app.goo.gl/Jvru1Qp3sDQALoL18',
            'rating': 4.5
    })

    elif predicted_class == 'Pantai Mandorak':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20mandorak&imgurl=https%3A%2F%2Ftripsumba.com%2Fwp-content%2Fuploads%2F2020%2F07%2FPantai-Mandorak-1-e1594876477547.jpg&imgrefurl=https%3A%2F%2Ftripsumba.com%2Fpantai%2Fpantai-mandorak-sumba-destinasi-pantai-tersembunyi-yang-menakjubkan-2%2F&docid=V2J0KN_rulNH8M&tbnid=0MXW2W8Uzuam7M&vet=12ahUKEwjak6OG6OGGAxVISmwGHchcA5UQM3oECHUQAA..i&w=650&h=420&hcb=2&ved=2ahUKEwjak6OG6OGGAxVISmwGHchcA5UQM3oECHUQAA',
            'artikel': ("Pantai Mandorak adalah sebuah destinasi wisata yang terletak di Desa Kalena Rongo, Kecamatan Kodi Utara, "
                        "Kabupaten Sumba Barat Daya, Nusa Tenggara Timur. Pantai ini dikenal dengan pasir putihnya dan batu karang. "
                        "Suasana pantai makin terasa alami dengan deburan ombak yang keras. Batu karang Pantai Mandorak saling berhimpit "
                        "membuat pandangan ke pantai cukup sempit karena tertutup tebing. Pantai ini memiliki pesona alam yang memikat "
                        "dengan pasir putih serta air laut yang jernih berwarna hijau toska. Pantai Mandorak juga dikelilingi oleh "
                        "perbukitan yang menambah keindahan alamnya. Pantai Mandorak dikenal dengan gelombang lautnya yang tinggi, "
                        "sehingga lebih cocok untuk bermain-main di air daripada berenang."),
            'lokasi': 'https://maps.app.goo.gl/2ZazDuVvtrfgUkHW',
            'rating': 4.5
    })

    elif predicted_class == 'Pantai Oetune':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20oetune&imgurl=https%3A%2F%2Fbobobox.com%2Fblog%2Fwp-content%2F%2Fuploads%2F2020%2F01%2Fwisataoke.jpg&imgrefurl=https%3A%2F%2Fbobobox.com%2Fblog%2Fpantai-oetune-gurun-nya-indonesia-yang-indah-di-pulau-sumba%2F&docid=Ot6EtFdMmmVyQM&tbnid=BWN-Ezy7GRJf2M&vet=12ahUKEwis9rGT6OGGAxVmdmwGHcEMC1YQM3oECGMQAA..i&w=1035&h=641&hcb=2&ved=2ahUKEwis9rGT6OGGAxVmdmwGHcEMC1YQM3oECGMQAA',
            'artikel': ("Pantai Oetune merupakan pantai yang sangat unik dan indah di Nusa Tenggara Timur (NTT). Pasalnya Pantai Oetune "
                        "memiliki pasir yang halus seperti pasir gurun. Tak hanya pasirnya yang halus pantai ini pun memiliki dentuman "
                        "ombak yang sangat kencang dan pecah di setiap gulungannya. Di tepi pantai ini pun dikelilingi banyaknya Pohon "
                        "Kasuari yang berusia puluhan tahun. Pantai Oetune ini berlokasi di Tuafanu, Kecamatan Kualin, Kabupaten Timor "
                        "Tengah Selatan, NTT. Akses menuju Pantai Oetune kurang lebih 115 kilometer dari pusat Kota Kupang dan jarak "
                        "tempuh sekitar 2-3 jam di perjalanan. Air laut yang ada di Pantai Oetune ini sangat biru dan bersih hingga Anda "
                        "dapat bercermin. Pantai Oetune memiliki pasir putih yang sangat halus dan menggunung membuat seolah-olah sedang "
                        "berada di gurun pasir."),
            'lokasi': 'https://maps.app.goo.gl/AxuQUF5Nd7BPkTXd9',
            'rating': 4.6
    })

    elif predicted_class == 'Pantai Waecicu':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20waecicu&imgurl=https%3A%2F%2Fmawatu.co.id%2Fwp-content%2Fuploads%2F2024%2F03%2FDaya-Tarik-Wisata-Pantai-Waecicu.webp&imgrefurl=https%3A%2F%2Fmawatu.co.id%2Fblog%2Fhealing-di-pantai-waecicu-labuan-bajo%2F&docid=y42ZW0QgpcuyPM&tbnid=HWfQPN1wNDklEM&vet=12ahUKEwiYh5Se6OGGAxV8TmwGHXbJBVQQM3oECGUQAA..i&w=750&h=450&hcb=2&ved=2ahUKEwiYh5Se6OGGAxV8TmwGHXbJBVQQM3oECGUQAA',
            'artikel': ("Pantai Waecicu adalah sebuah destinasi wisata yang terletak di Labuan Bajo, Komodo, Kabupaten Manggarai Barat, "
                        "Nusa Tenggara Timur. Pantai ini dikenal dengan pasir putihnya yang membentang sepanjang 1 km dan banyak bukit "
                        "hijau kecil di sekitarnya. Pantai ini memiliki tipe perairan yang dangkal dan berbagai pemandangan biota laut "
                        "yang cantik. Selain itu, Pantai Waecicu juga merupakan spot terbaik untuk menikmati matahari terbenam (sunset) "
                        "sambil duduk santai di jembatan yang digunakan oleh nelayan bersandar perahu."),
            'lokasi': 'https://maps.app.goo.gl/tdp3Ac6wTDxEFHjZA',
            'rating': 3.5
    })

    elif predicted_class == 'Pantai Walakiri':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20walakiri&imgurl=https%3A%2F%2Fsuperlive.id%2Fstorage%2Fsuperadventure%2F2018%2F05%2F25%2F58cf82e5a935.jpg&imgrefurl=https%3A%2F%2Fsuperlive.id%2Fsuperadventure%2Fartikel%2Fwilderness%2Fpantai-walakiri-titipan-ibu-pertiwi-di-negeri-sumba-beserta-6-fakta-uniknya&docid=Sl0B8mV-BDAr4M&tbnid=z5NSZtLwwlqMXM&vet=12ahUKEwidzvio6OGGAxUWSWwGHVTuCisQM3oECBgQAA..i&w=1058&h=570&hcb=2&ved=2ahUKEwidzvio6OGGAxUWSWwGHVTuCisQM3oECBgQAA',
            'artikel': ("Pantai Walakiri adalah surga pasir putih yang tersembunyi di Pulau Sumba, Nusa Tenggara Timur, Indonesia. "
                        "Terletak di Desa Walakiri, Kecamatan Kodi, pantai ini menawarkan pemandangan yang menakjubkan dengan pasir "
                        "putihnya yang halus, air laut yang biru jernih, dan batu-batu karang yang menjulang di sekitarnya. Salah "
                        "satu daya tarik utama Pantai Walakiri adalah pohon lontar yang ikonik yang tumbuh di sepanjang pantai dan "
                        "menjadi spot foto yang populer bagi para pengunjung. Selain menikmati keindahan alam, pengunjung juga dapat "
                        "bersantai, berenang, atau berjemur di Pantai Walakiri. Dengan suasana yang tenang dan keindahan alam yang "
                        "memukau, Pantai Walakiri adalah destinasi yang sempurna untuk melarikan diri dari keramaian dan menikmati "
                        "pesona alam Sumba yang masih alami."),
            'lokasi': 'https://maps.app.goo.gl/WSkR9tjEgAyRjbuj8',
            'rating': 4.6
    })

    elif predicted_class == 'Pantai Watu Parunu':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20watu%20parunu&imgurl=https%3A%2F%2Fwisatasumba.co.id%2Fwp-content%2Fuploads%2F2020%2F01%2FMenatap-Kesunyian-Tebing-Pantai-di-Watu-Parunu.jpg&imgrefurl=https%3A%2F%2Fwisatasumba.co.id%2Fmenatap-kesunyian-tebing-pantai-di-watu-parunu.html&docid=Nhg4Ab49JaCUhM&tbnid=VcDH_hMN3t5e7M&vet=12ahUKEwizxc266OGGAxXeSmwGHX2YBj0QM3oECEAQAA..i&w=855&h=570&hcb=2&ved=2ahUKEwizxc266OGGAxXeSmwGHX2YBj0QM3oECEAQAA',
            'artikel': ("Pantai Watu Parunu adalah destinasi wisata yang menawarkan pantai, perbukitan, dan tebing. Pantai ini "
                        "memiliki garis pantai yang sangat panjang dan berhubungan langsung dengan Samudra Indonesia, sehingga "
                        "ombaknya lumayan besar. Pengunjung yang datang ke Pantai Watu Parunu akan disambut hamparan pasir putih, "
                        "tebing batu, dan desir ombak yang datang silih berganti. Tebing batu di Pantai Watu Parunu berwarna "
                        "putih dan berada di ujung timur, menjadi salah satu ikon dari Pantai Watu Parunu yang selalu menjadi "
                        "spot untuk berfoto. Nama Watu Parunu sendiri berkaitan dengan fenomena alam batu berlubang yang menjadi "
                        "ciri khas lain dari pantai ini. Untuk bisa berjalan melewati batu berlubang ini, pengunjung harus merunduk."),
            'lokasi': 'https://maps.app.goo.gl/zaRNyMpVksxwfPJDA',
            'rating': 4.5
    })

    elif predicted_class == 'Pink Beach':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pink%20beach&imgurl=https%3A%2F%2Fasset-2.tstatic.net%2Fkupang%2Ffoto%2Fbank%2Fimages%2Frahasia-pink-beach-di-labuan-bajo-menyaksikan-indahnya-pasir-merah-muda.jpg&imgrefurl=https%3A%2F%2Fkupang.tribunnews.com%2F2021%2F01%2F03%2Frahasia-pink-beach-di-labuan-bajo-menyaksikan-indahnya-pasir-merah-muda%3Fpage%3Dall&docid=GIp9j46vcPyaVM&tbnid=tPU91xpd6jF4TM&vet=12ahUKEwjLhtLG6OGGAxWBSmwGHU_6AtUQM3oECBcQAA..i&w=700&h=393&hcb=2&ved=2ahUKEwjLhtLG6OGGAxWBSmwGHU_6AtUQM3oECBcQAA',
            'artikel': ("Pink Beach adalah salah satu pantai eksotis yang memiliki pemandangan indah dengan pasir pantai berwarna "
                        "merah muda atau pink. Pantai ini juga dikenal dengan nama Pantai Merah di Pulau Komodo. Melansir dari "
                        "Labuanbajotour, Pink Beach mempunyai warna pasir yang unik dan kerap menjadi destinasi wisata favorit "
                        "turis lokal dan mancanegara. Diketahui, warna pink pada pasirnya berasal dari hewan berukuran mikroskopis "
                        "bernama “Foraminifera” yang memberikan pigmen warna merah pada koral. Sehingga, koral-koral tersebut "
                        "terbawa oleh gelombang dan berakhir di pesisir pantai dan hancur menjadi serpihan serta butiran yang menjadi "
                        "pasir pantai. Sehingga, pantai ini pun disebut Pantai Pink yang memesona."),
            'lokasi': 'https://maps.app.goo.gl/JQCSuboXnnrAcS8p8',
            'rating': 4.8
    })

    elif predicted_class == 'Pulau Kalong':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pantai%20kalong&imgurl=https%3A%2F%2Fvisitingjogja.jogjaprov.go.id%2Fwp-content%2Fuploads%2F2021%2F03%2F56421040_133671817691055_1304561249744239884_n.jpg&imgrefurl=https%3A%2F%2Fvisitingjogja.jogjaprov.go.id%2F30423%2Fpantai-pulau-kalong-pulau-hijau-eksotis-yang-menantang-adrenalin%2F&docid=VYzlLRhRJxWO5M&tbnid=S14ageaNl2UhNM&vet=12ahUKEwjttPfP6OGGAxUuUGwGHSITDtEQM3oFCIIBEAA..i&w=1080&h=608&hcb=2&ved=2ahUKEwjttPfP6OGGAxUuUGwGHSITDtEQM3oFCIIBEAA',
            'artikel': ("Pulau Kalong, juga dikenal sebagai Kalong Padar atau Flying Fox Island, adalah sebuah pulau kecil yang "
                        "tidak berpenghuni dan terletak di Taman Nasional Komodo, dekat Labuan Bajo di provinsi Nusa Tenggara "
                        "Timur, Indonesia. Pulau ini dinamai setelah ribuan kelelawar yang menghuninya. Pulau ini menjadi salah "
                        "satu destinasi yang dikenal memiliki koloni kelelawar dengan jumlah besar yang mendiami pohon bakau di "
                        "pulau Kalong. Pulau ini adalah rumah bagi ribuan satwa liar yang masih eksis dan menggantungkan hidupnya "
                        "di sana, termasuk monyet, burung, hingga reptil. Selain itu, Pulau Kalong juga memiliki berbagai jenis "
                        "terumbu karang yang indah."),
            'lokasi': 'https://maps.app.goo.gl/zn1URtEZLpCAo1Ay9',
            'rating': 5
    })

    elif predicted_class == 'Pulau Kanawa':
        response.update({
            'image' : 'https://www.google.com/imgres?q=pulau%20kanawa&imgurl=https%3A%2F%2Fcdnwpseller.gramedia.net%2Fwp-content%2Fuploads%2F2024%2F06%2Fimage-23.jpeg&imgrefurl=https%3A%2F%2Fwww.gramedia.com%2Fbest-seller%2Fpulau-kanawa%2F&docid=XIdOhGuUm-DjdM&tbnid=INMgdkGBoK7giM&vet=12ahUKEwiypc3Z6OGGAxXkbmwGHTJhA_QQM3oECBoQAA..i&w=640&h=428&hcb=2&ved=2ahUKEwiypc3Z6OGGAxXkbmwGHTJhA_QQM3oECBoQAA',
            'artikel': ("Pulau Kanawa adalah sebuah pulau kecil yang terletak di perairan Flores, Nusa Tenggara Timur (NTT). "
                        "Pulau ini sangat terkenal karena keindahan bawah lautnya yang memukau. Ada banyak sekali terumbu "
                        "karang yang tumbuh di pulau ini dan itu menjadi daya tarik bagi wisatawan yang suka menyelam "
                        "sekaligus melihat kecantikan bawah laut Pulau Kanawa. Pulau ini memiliki luas sekitar 32 ha dan "
                        "jaraknya kurang lebih sekitar 15 km dari Labuan Bajo. Banyak yang bilang Pulau Kanawa Labuan Bajo "
                        "adalah gerbang dari Pulau Komodo. Pulau ini sangat cocok untuk kamu yang tidak "
                        "suka dengan tempat ramai."),
            'lokasi': 'https://maps.app.goo.gl/XphhXGxR5aqTDbRf9',
            'rating': 4
    })

    elif predicted_class == 'Rumah Budaya Sumba':
        response.update({
            'image' : 'https://www.google.com/imgres?q=rumah%20budaya%20sumba&imgurl=https%3A%2F%2Fassets-a1.kompasiana.com%2Fitems%2Falbum%2F2023%2F11%2F08%2Frumah-budaya-sumba4-654b101aee794a258a56ae22.jpg&imgrefurl=https%3A%2F%2Fwww.kompasiana.com%2Firwan11022%2F6548d957ee794a05bc7ddc72%2Fpetualangan-eksklusif-di-nusa-tenggara-timur-mengintip-keunikan-rumah-budaya-sumba-dan-pesona-budaya-lokal-yang-tersembunyi&docid=8LKd1_p_bjH4KM&tbnid=CiIP6_201Kgt1M&vet=12ahUKEwi7wPTl6OGGAxWNTmwGHcptBjgQM3oECBMQAA..i&w=1280&h=720&hcb=2&ved=2ahUKEwi7wPTl6OGGAxWNTmwGHcptBjgQM3oECBMQAA',
            'artikel': ("Rumah Budaya Sumba adalah museum khusus yang digunakan untuk memperkenalkan sejarah dan budaya "
                        "Sumba. Fungsi Rumah Budaya Sumba adalah sebagai museum sekaligus tempat wisata, penelitian, dan "
                        "pertemuan, serta pusat pembelajaran kebudayaan Sumba. Rumah Budaya Sumba mengoleksi berbagai "
                        "macam peninggalan kelompok etnik daerah Sumba yang berasal dari masa prasejarah hingga masa kini. "
                        "Koleksi-koleksi ini merupakan sumbangan koleksi pribadi Pater Robert Ramone dan sumbangan dari "
                        "setiap rumah adat Sumba. Rumah Budaya Sumba terletak di Jalan Rumah Budaya Nomor 212, Kalembu "
                        "Nga’banga Waitabula, Kabupaten Sumba Barat Daya, Provinsi Nusa Tenggara Timur."),
            'lokasi': 'https://maps.app.goo.gl/oKVbDfiYY7QPHAtP6',
            'rating': 3.5
    })

    elif predicted_class == 'Savana Puru Kambera':
        response.update({
            'image' : 'https://www.google.com/imgres?q=savana%20puru%20kambera&imgurl=https%3A%2F%2Findonesiajuara.asia%2Fwp-content%2Fuploads%2F2022%2F12%2FSavana-Puru-Kambera-4.webp&imgrefurl=https%3A%2F%2Findonesiajuara.asia%2Fblog%2Fsavana-puru-kambera-sumba%2F&docid=VGItbvKOh4MNAM&tbnid=UNSVXy7unVB3NM&vet=12ahUKEwj2n_ft6OGGAxW-SGwGHcG2BwkQM3oECBQQAA..i&w=1200&h=1200&hcb=2&ved=2ahUKEwj2n_ft6OGGAxW-SGwGHcG2BwkQM3oECBQQAA',
            'artikel': ("Savana Puru Kambera di Sumba Timur, Nusa Tenggara Timur, adalah rumah bagi kuda liar Sumba "
                        "dan menawarkan padang rumput yang luas dengan panorama indah. Terletak dekat dengan Pantai "
                        "Puru Kambera, sabana ini menjadi destinasi utama bagi pengunjung yang ingin melihat gerombolan "
                        "kuda liar. Dengan pemandangan ala Afrika dan latar belakang biru laut, Sabana Puru Kambera adalah "
                        "tempat yang menakjubkan untuk dinikmati."),
            'lokasi': 'https://maps.app.goo.gl/4ehKM9xiFN2rRMiP7',
            'rating': 4.6
    })

    elif predicted_class == 'Taman Nostalgia Kupang':
        response.update({
            'image' : 'https://www.google.com/imgres?q=taman%20nostalgia%20kupang&imgurl=https%3A%2F%2Fstatic.promediateknologi.id%2Fcrop%2F0x0%3A0x0%2F0x0%2Fwebp%2Fphoto%2Fp2%2F41%2F2024%2F02%2F05%2FIlustrasi-Taman-Nostalgia-2538799322.jpg&imgrefurl=https%3A%2F%2Fwww.kupangnews.com%2Fdaerah%2F414089034%2Fpemkot-segera-tata-ulang-taman-nostalgia&docid=rz_u7MKcTIPOzM&tbnid=yETICejE1fs8cM&vet=12ahUKEwi64KeB6eGGAxUBSmwGHau4DyMQM3oECCwQAA..i&w=742&h=466&hcb=2&ved=2ahUKEwi64KeB6eGGAxUBSmwGHau4DyMQM3oECCwQAA',
            'artikel': ("Taman Nostalgia Kupang adalah sebuah tempat yang unik dan bersejarah. Taman ini dikenal karena "
                        "menjadi rumah bagi Gong Perdamaian, sebuah gong raksasa yang menjadi simbol perdamaian dan "
                        "persatuan. Gong ini merupakan hadiah dari Pemerintah Tiongkok kepada Provinsi Nusa Tenggara Timur. "
                        "Selain Gong Perdamaian, taman ini juga memiliki replika bangunan-bangunan khas Nusa Tenggara Timur "
                        "dan area hijau yang luas. Pengunjung dapat menikmati suasana yang tenang dan berfoto di sekitar "
                        "taman. Taman Nostalgia Gong Perdamaian menjadi destinasi yang menarik bagi wisatawan yang ingin "
                        "menikmati keindahan alam dan belajar tentang budaya lokal."),
            'lokasi': 'https://maps.app.goo.gl/2HFMWe9NRaBPAJR39',
            'rating': 4.4
    })

    elif predicted_class == 'Wae Rebo Village':
        response.update({
            'image' : 'https://www.google.com/imgres?q=wae%20rebo&imgurl=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fd%2Fd5%2FWae_Rebo_di_Pagi_Hari.jpg%2F1200px-Wae_Rebo_di_Pagi_Hari.jpg&imgrefurl=https%3A%2F%2Fid.wikipedia.org%2Fwiki%2FWae_Rebo&docid=0C17FC5MCZ6N6M&tbnid=Qdi2fhwzDWLn9M&vet=12ahUKEwjbxMiQ6eGGAxVWTGwGHd64CE4QM3oECBUQAA..i&w=1200&h=793&hcb=2&ved=2ahUKEwjbxMiQ6eGGAxVWTGwGHd64CE4QM3oECBUQAA',
            'artikel': ("Wae Rebo adalah desa adat kecil yang terletak di pegunungan terpencil, Kampung Satar Lenda, "
                        "Kecamatan Satar Mese Barat, Kabupaten Manggarai, Nusa Tenggara Timur. Desa ini dikenal dengan 7 "
                        "rumah adatnya dan pemandangan khas pegunungan."),
            'lokasi': 'https://maps.app.goo.gl/oYo4uGq4E3ApMavy5',
            'rating': 4.7
    })

    elif predicted_class == 'Waikuri Lagoon':
        response.update({
            'image' : 'https://www.google.com/imgres?q=waikuri%20lagoon&imgurl=https%3A%2F%2Fauthentic-indonesia.com%2Fwp-content%2Fuploads%2F2020%2F06%2Fweekuri-lake-water-has-a-salty-taste.jpg&imgrefurl=https%3A%2F%2Fauthentic-indonesia.com%2Fblog%2Fhidden-paradise-weekuri-lagoon-in-sumba%2F&docid=FATlSo2hg-j-BM&tbnid=e5Hbb9a8QBs3WM&vet=12ahUKEwiziK-d6eGGAxW4RmwGHTouDsoQM3oECBgQAA..i&w=1127&h=845&hcb=2&ved=2ahUKEwiziK-d6eGGAxW4RmwGHTouDsoQM3oECBgQAA',
            'artikel': ("Waikuri Lagoon hadir dengan air yang sangat jernih dan berwarna biru kehijauan dengan kandungan "
                        "air asin dan payau. Ya, tidak seperti danau lainnya, air danau ini asin karena terbentuk dari "
                        "air lautan lepas yang ada di sekitar danau dan masuk dari celah bebatuan di gugusan karang "
                        "sekitarnya. Laguna ini dipagari batu karang alami di mana tumbuh pepohonan rimbun dan semak "
                        "belukar yang seolah menyembunyikan keindahannya dari kejauhan. Selain bermain di laguna, para "
                        "pengunjung juga bisa menyusuri jalan setapak ke arah barat dan mendaki bukit karang yang menjorok "
                        "ke lautan lepas dengan hati-hati. Dari atas bukit karang ini, pengunjung dapat menikmati pesona "
                        "Danau Weekuri dari ketinggian dan ombak lautan biru di sekitarnya yang menghantam gugusan karang "
                        "di sekitar danau."),
            'lokasi': 'https://maps.app.goo.gl/8MGi7iAQLrChGYen9',
            'rating': 4.7
    })

    elif predicted_class == 'Wisata Adat Kampung Todo':
        response.update({
            'image' : 'https://www.google.com/imgres?q=wisata%20adat%20kampung%20todo&imgurl=https%3A%2F%2Fasset.kompas.com%2Fcrop%2F0x167%3A1000x667%2F780x390%2Fdata%2Fphoto%2F2018%2F11%2F08%2F268456002.jpg&imgrefurl=https%3A%2F%2Ftravel.kompas.com%2Fread%2F2018%2F11%2F09%2F072941327%2Fkampung-adat-todo-pusat-peradaban-minangkabau-di-flores-barat%3Fpage%3Dall&docid=ZlTogjnXZP3S_M&tbnid=XEwl-diIRpvmpM&vet=12ahUKEwjc_veo6eGGAxW-cmwGHd-FB5wQM3oECGAQAA..i&w=780&h=390&hcb=2&ved=2ahUKEwjc_veo6eGGAxW-cmwGHd-FB5wQM3oECGAQAA',
            'artikel': ("Kampung Adat Todo adalah sebuah kampung adat yang terletak di Desa Todo, Kecamatan Satar Mese, "
                        "Kabupaten Manggarai, Pulau Flores, Provinsi Nusa Tenggara Timur. Kampung ini dikenal sebagai kampung "
                        "adat tertua di Manggarai Raya dan memiliki bentuk bangunan rumah adat yang khas serta lingkungan "
                        "perkampungan adat yang indah. Kampung Adat Todo juga dikenal sebagai pusat peradaban Minangkabau. "
                        "Selain itu, kampung ini juga merupakan pusat kerajaan Manggarai di zaman dulu. Di sini, wisatawan "
                        "dapat melakukan berbagai aktivitas seperti merasakan bagaimana tamu disambut dengan cara adat orang "
                        "Manggarai, mengenakan pakaian adat keseharian masyarakat Todo, belajar bertenun dengan ibu-ibu yang "
                        "menenun di depan rumah serta belajar sejarah mengenai kampung Todo dan kerajaan Todo."),
            'lokasi': 'https://maps.app.goo.gl/GXqcJSFkqFC9q9Xd8',
            'rating': 3.5
    })

    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8002))
    app.run(host='0.0.0.0', port=port)