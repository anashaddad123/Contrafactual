import glob
list_wav = glob.glob("/home/nas-wks01/users/uapv2202503/testtp/data_folder/data1"+"/**/*.wav",recursive=True)
#fichier_spk2 = open("spk2utt", "w")
fichier_utt = open("utt2spk", "w")
fichier_wave = open("wav.scp", "w")

for idx,i in enumerate(list_wav):
    print(idx)
    #fichier_spk2.write("spk-{} spk-{}_utt-{}\n".format(idx,idx,idx))
    fichier_utt.write("spk-{}_utt-{} spk-{}\n".format(idx,idx,idx))
    fichier_wave.write("spk-{}_utt-{} {}\n".format(idx,idx,i))


