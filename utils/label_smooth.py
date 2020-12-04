#method
#criterion = Label_smoothing(num_class=2,eps=0.1)

class Label_smoothing(nn.Module):
    def __init__(self, eps=0.1, num_class=1000,reduction='mean',loss_type=None):
        super(Label_smoothing, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_type = loss_type
        self.focal_loss = FocalLoss()
        self.margin_loss = LargeMarginInSoftmaxLoss()
    def forward(self, output, target):
        c = 2
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        if self.loss_type==None:
            return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
        elif self.loss_type=="focal":
            return loss*self.eps/c + (1-self.eps)*self.focal_loss(output,target)
        elif  self.loss_type="margin":
            return loss*self.eps/c + (1-self.eps)*self.margin_loss(output,target)
