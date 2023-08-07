# Generating Audio Samples with Machine Learning
 
I started this project in 2020, paused development for a couple years, then revisited it in 2023. It went through a few different iterations.

Initial, naive implementation: generate raw wav text with GPT-2
	- this doesn’t work well, generally doesn’t yield playable audio

Next iteration: Approach as linear regression problem - train an RNN (with LSTM) to predict next pressure position in a sequence.
	- this yields playable audio, but it generally sounds like white noise and doesn’t learn well with the limited resources I tried allocating to the task.


Current iteration: In the WaveNet paper, they found that bucketizing the pressure positions into bins and approaching as a classification problem worked better than attempting linear regression, since learning pressure bins independently avoids potentially unhelpful assumptions inherent to linear regression.

Informed by this strategy, I trained GPT-2 to predict the next bin in a sequence of binned pressure positions. This approach enables the model to learn real patterns inherent to the source material it’s trained on.