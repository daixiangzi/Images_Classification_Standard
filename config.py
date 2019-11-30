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
        #label_smoothing
        label_smooth = False
        esp = 0.1
        # warmming_up
        warmming_up = False
        decay_epoch=1000
        #mix up
        mix_up= True
        alpha = 0.5#0.1

        # Cutout
        cutout = False #set cutout flag
        cutout_n = 5
        cutout_len = 5

        evaluate=False #wether to test
        start_epoch = 0
        optim = "SGD" #SGD,Adam,RAdam,AdamW
        #lookahead
        lookahead = False
        la_steps=5 
        la_alpha=0.5
        logs = './logs/'+arch+str(depth)+optim+str(lr)+("_lookahead" if lookahead else "")+"_"+activation+"_"+init+("_cutout" if cutout else "")+("_mix_up"+str(alpha) if mix_up else "")+("_warmup"+str(decay_epoch) if warmming_up else "")+str('_label_smooth'+str(esp) if label_smooth else "")
