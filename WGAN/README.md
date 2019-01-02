### Purpose

<p> This is an implementation of WGAN with weight clipping. </p>
<p> For more information, visit: </p>
<p> https://arxiv.org/abs/1701.07875 </p>
<p> https://github.com/eriklindernoren/Keras-GAN/tree/master/wgan </p>

### Problems to be considered

Notice that the result of WGAN is not satisfying compared to GAN result.
I believe this comparably unsatisfying result is caused by the difficulty of hyperparametrizing weight clipping value.
Thus, I'm planning to apply WGAN-GP (gradient-penalty) method to improve the result.
