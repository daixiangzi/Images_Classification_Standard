class Config(object):
        data = './'
        activation='Relu'#Swish,Relu,Mish,Selu
        init = "kaiming"#kaiming
        save = './checkpoints'#save best model dir
        arch = 'resnet'
        depth = 50 #resnet-50
        gpu_id = '0,1' #gpu id
        train_data = '/home/daixiangzi/dataset/cifar-10/files/train.txt'# train file
        test_data = '/home/daixiangzi/dataset/cifar-10/files/test.txt'# test file
        train_batch=512 
        test_batch=512
        epochs= 150
        lr = 0.1#0.003
        gamma =0.1#'LR is multiplied by gamma on schedule.
        drop = 0
        momentum = 0.9
        fre_print=2
        weight_decay = 1e-4
        schedule = [60,100,125]
        seed = 666
        workers=4
        num_classes=10 #classes
        resume = None #path to latest checkpoint
        # warmming_up
        warmming_up = False
        decay_epoch=1000

        # Cutout
        cutout = None #set cutout flag
        cutout_n = 5
        cutout_len = 5

        evaluate=False #wether to test
        start_epoch = 0
        optim = "SGD" #SGD,Adam,RAdam,AdamW
        #lookahead
        lookahead = False
        la_steps=5 
        la_alpha=0.5
        logs = './logs/'+arch+str(depth)+optim+str(lr)+"_"+activation+"_"+init
