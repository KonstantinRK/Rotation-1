from data_wrapper import RawData


ResCal = [RawData("ResNet50_ILSVRC_cal_data.csv", "ResCal")]
ResVal = [RawData("ResNet50_ILSVRC_val_data.csv", "ResVal")]
ResAwA = [RawData("ResNet50_AwA_data.csv", "ResAwA")]
ResD = [RawData("ResNet50_ILSVRC_cal_data+dead_percent0.01.csv", "ResD1"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.05.csv", "ResD5"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.1.csv", "ResD10"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.2.csv", "ResD20"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.3.csv", "ResD30"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.4.csv", "ResD40"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.5.csv", "ResD50"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.6.csv", "ResD60"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent0.8.csv", "ResD80"),
        RawData("ResNet50_ILSVRC_cal_data+dead_percent1.0.csv", "ResD100")]

ResN =[RawData("ResNet50_ILSVRC_cal_data+noise_sigma5.csv", "ResN5"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma10.csv", "ResN10"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma15.csv", "ResN15"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma20.csv", "ResN20"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma30.csv", "ResN30"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma50.csv", "ResN50"),
       RawData("ResNet50_ILSVRC_cal_data+noise_sigma100.csv", "ResN100")]

SquCal = [RawData("SqueezeNet_ILSVRC_cal_data.csv", "SuqCal")]
SquVal = [RawData("SqueezeNet_ILSVRC_val_data.csv", "SuqVal")]
SquAwA = [RawData("SqueezeNet_AwA_data.csv", "SuqAwA")]
SquD = [RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.01.csv", "SuqD1"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.05.csv", "SuqD5"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.1.csv", "SuqD10"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.2.csv", "SuqD20"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.3.csv", "SuqD30"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.4.csv", "SuqD40"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.5.csv", "SuqD50"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.6.csv", "SuqD60"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent0.8.csv", "SuqD80"),
        RawData("SqueezeNet_ILSVRC_cal_data+dead_percent1.0.csv", "SuqD100")]

SquN =[RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma5.csv", "SuqN5"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma10.csv", "SuqN10"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma15.csv", "SuqN15"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma20.csv", "SuqN20"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma30.csv", "SuqN30"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma50.csv", "SuqN50"),
       RawData("SqueezeNet_ILSVRC_cal_data+noise_sigma100.csv", "SuqN100")]


ResOOS = ResAwA + ResD + ResN
SquOOS = SquAwA + SquD + SquN


ResTest = ResVal + ResOOS
SquTest = SquVal + SquOOS

ResData = ResCal + ResTest
SquData = SquCal + SquTest

AllData = ResData + SquData


MobCal = [RawData("MobileNet25_ILSVRC_cal_data.csv", "MobCal")]
MobVal = [RawData("MobileNet25_ILSVRC_val_data.csv", "MobVal")]
MobAwA = [RawData("MobileNet25_AwA_data.csv", "MobAwA")]
MobD = [RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.01.csv", "MobD1"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.05.csv", "MobD5"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.1.csv", "MobD10"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.2.csv", "MobD20"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.3.csv", "MobD30"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.4.csv", "MobD40"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.5.csv", "MobD50"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.6.csv", "MobD60"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent0.8.csv", "MobD80"),
        RawData("MobileNet25_ILSVRC_cal_data+dead_percent1.0.csv", "MobD100")]

MobN =[RawData("MobileNet25_ILSVRC_cal_data+noise_sigma5.csv", "MobN5"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma10.csv", "MobN10"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma15.csv", "MobN15"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma20.csv", "MobN20"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma30.csv", "MobN30"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma50.csv", "MobN50"),
       RawData("MobileNet25_ILSVRC_cal_data+noise_sigma100.csv", "MobN100")]


MobOOS = MobAwA + MobD + MobN
MobTest = MobVal + MobOOS
MobData = MobCal + MobTest