---
layout: post
title:  "Model Evolution - Machine Learning Pipeline"
desc: "Introduction of SigVoiced and a little bit of motivation to follow the posts."
keywords: "sigvoiced,machine learning,intro"
date: 02-07-2016
categories: [Sigvoiced]
tags: [machine learning,pattern recognition,classification,myo armband,hidden markov models,LSTM,SVM]
icon: fa-signing
---

<h1>Prerequisites</h1>
<a href="https://en.wikipedia.org/wiki/Machine_learning">Machine Learning</a> basics: A basic understanding should be enough.
<h1>The Pipeline</h1>
Solving a real world problem using machine learning is not so trivial. If done haphazardly, it may lead to a disaster or waste of a lot of effort. I have seen people directly jumping into choosing a classification algorithm for solving a pattern recognition task and ending up with an accuracy of 20%. Problem solving does not mean selecting an algorithm to solve a problem. It needs proper analysis and understanding  of the problem itself. So, my dear friends, for making sure that you do not end up banging your head on the wall saying "<em><strong>I wasted a lot of time for nothing</strong></em>" we have <strong>The Pipeline. </strong>This might sound a little boring and is a "<strong>Not So Fun</strong>" part of Machine Learning, but believe me, it is highly necessary for getting better results. Remember the bigger picture that I had talked about? If not then <a href="https://sigvoiced.wordpress.com/about/">read this</a>.
<h2>Understanding The Problem</h2>
<ol>
	<li><strong>What kind of problem are you trying to solve?</strong> <strong>:</strong> ask this question to yourself before jumping into anything. The following are a few examples,
<ul>
	<li><a href="https://en.wikipedia.org/wiki/Statistical_classification">Classification Problem</a> :  Where you want to classify data into different classes.</li>
	<li><a href="https://en.wikipedia.org/wiki/Regression_analysis">Regression Problem </a>: Where you want to predict some results given some observations.</li>
	<li><a href="https://en.wikipedia.org/wiki/Feature_extraction">Feature Extraction</a>: Extracting features from a given dataset that best represents your data.</li>
	<li><a href="https://en.wikipedia.org/wiki/Information_retrieval">Information Retrieval</a>: Where you have a set of documents and you want to retrieve the most relevant documents given a search query.</li>
	<li>And much more...</li>
</ul>
</li>
	<li><strong>Can the problem be divided? : </strong> After classifying the problem, determine whether you can divide the problem into subproblems or not. If you can, then do the same for every subproblem too.</li>
	<li><strong>How does the environment look like? : </strong>By environment, I mean the environment of your system. The following are a few questions that you can ask yourself to understand the environment better,
<ul>
	<li>Is the data accessible to the system?</li>
	<li>Is the data that the system gets continuous or discrete?</li>
	<li>How static or dynamic is your data.</li>
	<li>Is your data continuous or discrete?</li>
	<li>For details about environments from an Artificial Intelligence point of view, <a href="http://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_agents_and_environments.htm">read this</a>.</li>
</ul>
</li>
</ol>
For example, If my problem is "<strong>Recognising sign language using <a href="https://www.myo.com/">Myo armband</a> and converting that into speech</strong>" then
<ol>
	<li>The problem is a <strong>Classification Problem </strong>as I want to classify different <strong>hand gestures</strong> into a set of signs in a <strong>sign language</strong>.</li>
	<li>The <strong>Environment</strong> is,
<ul>
	<li><strong>Accessible : </strong>Time series data from EMG Pods, accelerometer and gyroscope of the Myo Armband.</li>
	<li><strong>Dynamic: </strong> The data from the sensors change dynamically.</li>
	<li><strong>Continuous : </strong> The data is time series, which means that we end up getting a sequence of data samples over time.</li>
	<li>And <strong>partially observable : </strong> Not everything is accessible from these sensors like fine finger movements.</li>
</ul>
</li>
</ol>
<h2>The Process</h2>
Once you have classified your problem and environment, you can follow a two-step process to obtain the best model for your system.
<h3>Step 1 : Data Visualization and Analysis</h3>
This is just an analysis step that will help in better understanding your data and later will help us in determining which Machine Learning Algorithm you should use.
<ol>
	<li><strong>Collect sample data</strong> from the environment for analysis. DO NOT collect all the data now. You just need a few samples for analysis.</li>
	<li><strong>Visualise the data</strong> to understand how it looks like and how it is distributed.</li>
	<li>Try to find out<strong> unique characteristics</strong> in your data. These characteristics can be features or combination of features.</li>
	<li>Determine what and how much <strong>preprocessing</strong> is required to clean and normalise the data.</li>
	<li>Determine if you can have<strong> labelled data</strong> for training, in the case of a classification problem.</li>
	<li><strong>Think of algorithms</strong> that can help in getting the desired result given such a data.</li>
</ol>
<em>Depending on whether you can have labelled or unlabeled data for training, you can decide whether you would use a supervised or an unsupervised learning algorithm. For more details <a href="http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/">read this</a>.</em>

<strong>Continuing the example</strong> of converting sign language to speech, the following are two different visualisations of the time series data that was acquired from the Myo Armband.

<img class="alignnone  wp-image-233" src="https://sigvoiced.files.wordpress.com/2016/07/screenshot-462.png" alt="Screenshot (46)" width="680" height="399" />

<img class="alignnone  wp-image-232" src="https://sigvoiced.files.wordpress.com/2016/07/screenshot-452.png?w=680" alt="Screenshot (45)" width="679" height="398" />

By following Step 1 we could,
<ol>
	<li>Understand the <strong>difference between the data from two different users</strong>.</li>
	<li><strong>Think of normalisation and scaling techniques</strong> for preprocessing the data.</li>
	<li><strong>Think of a set of features</strong> that we could extract from the data.</li>
	<li>Distinguish<strong> variant and invariant features</strong>.</li>
	<li>Realise that we could <strong>collect labelled data</strong> by conducting experiments.</li>
	<li>Think of <strong>Machine Learning Algorithms</strong> that might work in our case.
<ul>
	<li><strong>HMM</strong> (Hidden Markov Models)</li>
	<li><strong>SVM</strong> (Support Vector Machine)</li>
	<li><strong>LSTM Networks</strong> (Long Short Term Memory Networks)</li>
</ul>
</li>
</ol>
<h3>Step 2 : The Machine Learning Pipeline</h3>
<ol>
	<li><strong>Data Acquisition :  </strong> Acquire the training data</li>
	<li><strong>Data Preprocessing : </strong>The following are a few examples of preprocessing,
<ul>
	<li><strong><a href="https://en.wikipedia.org/wiki/Feature_scaling">Scaling</a></strong>: Scaling the data is highly necessary for standardising the model. In the next post, I will explain feature scaling with an example.</li>
	<li><strong>Normalisation:</strong> normalising the data by resampling or filtering</li>
</ul>
</li>
	<li><strong><a href="https://en.wikipedia.org/wiki/Feature_extraction">Feature Extraction </a>: </strong> Extract features from the data that represent your data well.</li>
	<li><strong>Feature Scaling: </strong>If you intend to use<a href="https://en.wikipedia.org/wiki/Maximum_likelihood_estimation" target="_blank"> Maximum likelihood estimation</a> for parameter estimation or optimisation algorithm like <a href="https://en.wikipedia.org/wiki/Gradient_descent">Gradient Descent</a> for optimising your parameters then scaling the features is highly necessary so that you arrive at the best parameters faster. Dr Andrew Ang has explained it in a very subtle way <a href="https://www.youtube.com/watch?v=jV7Zk5ri3Es" target="_blank">here</a>.</li>
	<li> <strong>Model Generation : </strong> Use any machine learning algorithm (Supervised or Unsupervised) to generate a model for your system.</li>
	<li><strong>Evaluation : </strong> Evaluate your model using k-fold-cross-validation (There are many other but this works the best for me. So, keep experimenting) and a few standard evaluation metrics like,
<ul>
	<li>Precision</li>
	<li>Recall</li>
	<li>Contingency Matrix</li>
	<li>Anova</li>
	<li>ROC</li>
	<li>F-Measure</li>
</ul>
</li>
</ol>
I will continue our example of<em><strong> converting sign language to speech</strong> </em>for step-1 and 2 in my upcoming posts. Do not get overwhelmed by the terms that you have read above. I will cover everything in step 1 and 2 in detail one by one.
<h1>Why are we doing all this?</h1>
The answer is pretty simple. To assure that we have a highly optimal model at the end for solving our initial problem.
<h1>Why the Sign Language To Speech example</h1>
This was my research project and it extensively covers every step in the entire machine learning pipeline . I learnt a lot from this project and I think it can help you folks in understanding a real-world problem and how Machine Learning can be used to solve it. Plus I think that it is a really cool application of Machine Learning.
<h1>What Next?</h1>
In my upcoming posts, I will break the example down as mentioned above and show the challenges that you might face and how to solve them. So if you have a few questions then I would ask you to be patient and go through the links for getting a better understanding. I will cover every point of  step-1 and step-2 in detail in individual posts so that you get a clear idea about how things work in the Machine Learning world.