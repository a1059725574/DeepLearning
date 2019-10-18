import tensorflow as tf


class ModifiedUNet():

    def __init__(self):

        self.img_height = 256
        self.img_width = 256
        self.img_channel = 3
        self.batch_size = 1    

    def discriminator(self, x, mask):
        
        # img
        img = x*mask    
        Layer1_1 = Conv2D(img, channels = 64, kernel_size = 4, stride = 2)
        Layer1_1 = LeakyRelu(Layer1_1)

        Layer2_1 = Conv2D(Layer1_1, channels = 128, kernel_size = 4, stride = 2)
        Layer2_1 = NormFormat(Layer2_1)
        Layer2_1 = LeakyRelu(Layer2_1)

        Layer3_1 = Conv2D(Layer2_1, channels = 256, kernel_size = 4, stride = 2)
        Layer3_1 = NormFormat(Layer3_1)
        Layer3_1 = LeakyRelu(Layer3_1)

        Layer4_1 = Conv2D(Layer3_1, channels = 512, kernel_size = 4, stride = 2)
        Layer4_1 = NormFormat(Layer4_1)
        Layer4_1 = LeakyRelu(Layer4_1)

        Layer5_1 = Conv2D(Layer4_1, channels = 512, kernel_size = 4, stride = 2)
        Layer5_1 = NormFormat(Layer5_1)
        Layer5_1 = LeakyRelu(Layer5_1)

        Layer6_1 = Conv2D(Layer5_1, channels = 512, kernel_size = 4, stride = 2)
        Layer6_1 = NormFormat(Layer6_1)
        Layer6_1 = LeakyRelu(Layer6_1)
    
        # hole
        hole = x*(1 - mask)
        Layer1_2 = Conv2D(hole, channels = 64, kernel_size = 4, stride = 2)
        Layer1_2 = LeakyRelu(Layer1_2)

        Layer2_2 = Conv2D(Layer1_2, channels = 128, kernel_size = 4, stride = 2)
        Layer2_2 = NormFormat(Layer2_2)
        Layer2_2 = LeakyRelu(Layer2_2)

        Layer3_2 = Conv2D(Layer2_2, channels = 256, kernel_size = 4, stride = 2)
        Layer3_2 = NormFormat(Layer3_2)
        Layer3_2 = LeakyRelu(Layer3_2)

        Layer4_2 = Conv2D(Layer3_2, channels = 512, kernel_size = 4, stride = 2)
        Layer4_2 = NormFormat(Layer4_2)
        Layer4_2 = LeakyRelu(Layer4_2)

        Layer5_2 = Conv2D(Layer4_2, channels = 512, kernel_size = 4, stride = 2)
        Layer5_2 = NormFormat(Layer5_2)
        Layer5_2 = LeakyRelu(Layer5_2)

        Layer6_2 = Conv2D(Layer5_2, channels = 512, kernel_size = 4, stride = 2)
        Layer6_2 = NormFormat(Layer6_2)

        Layer7 = tf.concat([Layer6_1, Layer6_2], axis = -1)
        output_logit = Conv2D(Layer7, channels = 1, kenel_size = 4, stride = 1)
        return output_logit

    def UNet(self, img, M_in):

        Layer1_1 = Conv2D(x, channels = 64, kernel_size = 4, stride = 2)
        layer1_2 = Conv2D(M_in, channels = 64, kernel_size = 4, stride = 2)
        Layer1_1 = tf.multiply(Layer1_1, gA(layer1_2))
        Layer1_1 = LeakyRelu(Layer1_1)

        Layer2_1 = Conv2D(Layer1_1, channels = 128, kernel_size = 4, stride = 2)
        layer2_2 = gM(layer1_2)
        layer2_2 = Conv2D(layer2_2, channels = 128, kernel_size = 4, stride = 2)
        Layer2_1 = tf.multiply(Layer2_1, gA(layer2_2))
        Layer2_1 = NormFormat(Layer2_1)
        Layer2_1 = LeakyRelu(Layer2_1)

        Layer3_1 = Conv2D(Layer2_1, channels = 256, kernel_size = 4, stride = 2)
        layer3_2 = gM(layer2_2)
        layer3_2 = Conv2D(layer3_2, channels = 256, kernel_size = 4, stride = 2)
        Layer3_1 = tf.multiply(Layer3_1, gA(layer3_2))
        Layer3_1 = NormFormat(Layer3_1)
        Layer3_1 = LeakyRelu(Layer3_1)

        Layer4_1 = Conv2D(Layer3_1, channels = 512, kernel_size = 4, stride = 2)
        layer4_2 = gM(layer3_2)
        layer4_2 = Conv2D(layer4_2, channels = 512, kernel_size = 4, stride = 2)
        Layer4_1 = tf.multiply(Layer4_1, gA(layer4_2))
        Layer4_1 = NormFormat(Layer4_1)
        Layer4_1 = LeakyRelu(Layer4_1)

        Layer5_1 = Conv2D(Layer4_1, channels = 512, kernel_size = 4, stride = 2)
        layer5_2 = gM(layer4_2)
        layer5_2 = Conv2D(layer5_2, channels = 512, kernel_size = 4, stride = 2)
        Layer5_1 = tf.multiply(Layer5_1, gA(layer5_2))
        Layer5_1 = NormFormat(Layer5_1)
        Layer5_1 = LeakyRelu(Layer5_1)

        Layer6_1 = Conv2D(Layer5_1, channels = 512, kernel_size = 4, stride = 2)
        layer6_2 = gM(layer5_2)
        layer6_2 = Conv2D(layer6_2, channels = 512, kernel_size = 4, stride = 2)
        Layer6_1 = tf.multiply(Layer6_1, gA(layer6_2))
        Layer6_1 = NormFormat(Layer6_1)
        Layer6_1 = LeakyRelu(Layer6_1)

        Layer7_1 = Conv2D(Layer6_1, channels = 512, kernel_size = 4, stride = 2)
        layer7_2 = gM(layer6_2)
        layer7_2 = Conv2D(layer7_2, channels = 512, kernel_size = 4, stride = 2)
        Layer7_1 = tf.multiply(Layer7_1, gA(layer7_2))
        Layer7_1 = NormFormat(Layer7_1)
        Layer7_1 = LeakyRelu(Layer7_1)

        # reverse mask
        reverse_Mask = 1.0 - M_in
        Layer1_3 = Conv2D(reverse_Mask, channels = 64, kernel_size = 4, stride = 2)
        
        Layer2_3 = gM(Layer1_3)
        Layer2_3 = Conv2D(Layer2_3, channels = 128, kernel_size = 4, stride = 2)
        
        Layer3_3 = gM(Layer2_3)
        Layer3_3 = Conv2D(Layer3_3, channels = 256, kernel_size = 4, stride = 2)

        Layer4_3 = gM(Layer3_3)
        Layer4_3 = Conv2D(Layer4_3, channels = 512, kernel_size = 4, stride = 2)

        Layer5_3 = gM(Layer4_3)
        Layer5_3 = Conv2D(Layer5_3, channels = 512, kernel_size = 4, stride = 2)

        Layer6_3 = gM(Layer5_3)
        Layer6_3 = Conv2D(Layer6_3, channels = 512, kernel_size = 4, stride = 2)

        # forward
        Layer8_1 = Conv2D(Layer7_1, channels = 512, kernel_size = 4, stride = 2) 
        Layer8_1 = tf.multiply(tf.concat([Layer8_1, Layer6_1], axis = -1), tf.concat([gA(Layer6_3), gA(Layer6_2)], axis = -1))
        Layer8_1 = NormFormat(Layer8_1)
        Layer8_1 = LeakyRelu(Layer8_1)
        
        Layer9_1 = Conv2D(Layer8_1, channels = 512, kernel_size = 4, stride = 2) 
        Layer9_1 = tf.multiply(tf.concat([Layer9_1, Layer5_1], axis = -1), tf.concat([gA(Layer5_3), gA(Layer5_2)], axis = -1))
        Layer9_1 = NormFormat(Layer9_1)
        Layer9_1 = LeakyRelu(Layer9_1)   

        Layer10_1 = Conv2D(Layer9_1, channels = 512, kernel_size = 4, stride = 2) 
        Layer10_1 = tf.multiply(tf.concat([Layer10_1, Layer4_1], axis = -1), tf.concat([gA(Layer4_3), gA(Layer4_2)], axis = -1))
        Layer10_1 = NormFormat(Layer10_1)
        Layer10_1 = LeakyRelu(Layer10_1)

        Layer11_1 = Conv2D(Layer10_1, channels = 256, kernel_size = 4, stride = 2) 
        Layer11_1 = tf.multiply(tf.concat([Layer11_1, Layer3_1], axis = -1), tf.concat([gA(Layer3_3), gA(Layer3_2)], axis = -1))
        Layer11_1 = NormFormat(Layer11_1)
        Layer11_1 = LeakyRelu(Layer11_1)

        Layer12_1 = Conv2D(Layer11_1, channels = 128, kernel_size = 4, stride = 2) 
        Layer12_1 = tf.multiply(tf.concat([Layer12_1, Layer2_1], axis = -1), tf.concat([gA(Layer2_3), gA(Layer2_2)], axis = -1))
        Layer12_1 = NormFormat(Layer12_1)
        Layer12_1 = LeakyRelu(Layer12_1)     

        Layer13_1 = Conv2D(Layer12_1, channels = 64, kernel_size = 4, stride = 2) 
        Layer13_1 = tf.multiply(tf.concat([Layer13_1, Layer1_1], axis = -1), tf.concat([gA(Layer1_3), gA(Layer1_2)], axis = -1))
        Layer13_1 = NormFormat(Layer13_1)
        Layer13_1 = LeakyRelu(Layer13_1)    

        Layer14_1 = Conv2D(Layer14_1, channels = 3, kernel_size = 4, stride = 2)
        output = Tanh(Layer14_1)
        return output

    def build_model(self):

        self.img = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_channel])
        Conv2D(self.)
        

        return 

    def gM(self):
        
        

        return 

    def gA(self):
    
        return 

    def GaussianAct(a = 1.1, mu = 2.0, sigma1 = 1.0, sigma2 = 1.0):
        
        a = tf.clip_by_value(a, min = 1.01, max = 6.0)
        mu = tf.clip_by_value(mu, min = 0.1, max = 3.0)
        sigma1 = tf.clip_by_value(sigma1, min = 1.0, max = 2.0)
        sigma2 = tf.clip_by_value(sigma2, min = 1.0, max = 2.0)
        return


if __name__ == "__main__":

    m_unet = ModifiedUNet()


