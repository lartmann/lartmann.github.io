---
layout: page
title: Sampling Equations From Variational Autoencoder Embeddings A Novel Approach to Symbolic Regression
description: Development of new approach to symbolic regression by using principles of Bayesian linear regression in combination with a variational autoencoder to fit non-linear functions additionally.
img: assets/img/VAE.drawio.png
importance: 1
related_publications: true
category: individual
toc:
  sidebar: left
---

> # Abstract
>
> The contribution investigates a new approach to symbolic regression that uses the principles of Bayesian linear regression to fit nonlinear functions additionally. In the future, this may enable hierarchical modeling that encodes individual differences in latent space and model architecture. To fill this methodological gap, the embedding of a (variational) autoencoder is used to perform a continuous relaxation of the discrete search space of functions and sample equations. Different architectures and model parameters are tested to find those that best satisfy the properties of continuity, semantic ordering of the equations, and searchability. The two main architectures have the difference that one uses n-hot encoded equation elements while the other implements one-hot encoded function terms. Both of which are implemented as standard and variational autoencoder.
> From the current results, a difference in performance among these two architectures can be observed. In detail, the autoencoders with n-hot encoded equation elements did not satisfy the continuity property as they produced impossible equations and thus resulted in a sparse space. However, the autoencoders with one-hot encoded function terms produced an embedding that satisfied all of the requirements. Finally, it is shown that this approach can work in principle, as it was possible to derive equations from the data using MCMC sampling.

# Introduction

Symbolic regression is a subfield of artificial intelligence that aims to derive closed-form mathematical formulae from observed data {% cite eq_discovery_theory %}.
This means that it takes the observed data as input and outputs the mathematical formula that best describes these data {% cite sym_reg_1 Todorovski2017 %}. In this instance, best refers to a well-balanced trade-off between simplicity and accuracy {% cite wang_symbolic_2019 %}.

There is the possibility of using trained neural networks as models, but understanding and insight are difficult to achieve. Symbolic regression models can learn longer time horizons compared to neural network models because they derive the underlying principle and are therefore more robust to noise. {% cite sindyc %}
For these reasons, symbolic regression algorithms are optimal for identifying appropriate mathematical equations that can be further transformed and interpreted to gain insight {% cite sym_reg_2 %}.

For scientific purposes, the relationship between the variables is often more interesting than the prediction alone. This is why symbolic regression is already used in physics with symbolic regression algorithms such as AI Feynman {% cite ai_feyman %}, which uses neural networks to find equations from the field of physics.
However, a variety of algorithms has been adopted to discover the most appropriate equations. For example, symbolic regression using genetic programming {% cite SR_symbolic_reg %}, or deep symbolic regression uses a gradient-based approach based on reinforcement learning. {% cite petersen2021deep %}

Despite the success of symbolic regression algorithms in the domain of physics, they have not been widely implemented in the behavioral and brain sciences such as psychology or neuroscience. Human sciences present challenges that are not as present in other domains. One such challenge is the large individual differences between subjects. This means that if each participant contributes multiple data points, then trying to find a linear regression line for all participants at once could violate the assumption of independence since the data are dependent on the individual. For this reason, hierarchical approaches such as Bayesian hierarchical linear regression play a pivotal role in these areas. They allow for modeling at the individual and group level simultaneously and can be seen as a generalization of linear models. {% cite multilevel individual_diff %}

The overarching objective of this thesis is to introduce a novel approach to symbolic regression that extends Bayesian linear regression to implement nonlinear equations\footnote{Functions can be understood as a form of equations. Equations referred to in the context of the implementation described in this thesis are also functions as they map an independent variable to a dependent variable.} and can, in the future, be further expanded for Bayesian hierarchical regression.

Specifically, this thesis focuses on developing a neural network architecture that creates a continuous space of equations that sampling algorithms can search to identify the most appropriate equation for the given data. This procedure relies on an autoencoder that represents semantic similarity by vector similarity in the embedding, also called latent space. This embedding can transform the discrete search space of function terms into a continuous search space of real numbers. This enables the use of Markov chain Monte Carlo sampling to fit an equation and obtain its underlying probability distribution in the latent space. Consequently, the discrete search space of equations is transformed into a continuous search space by using continuous relaxation.
Hence, there are three main properties that the search space needs to fulfill:

- Continuity
- Representation of Semantic Similarity
- Searchability

Finally, a proof of principle is presented showing that it is indeed possible to search this space and find an equation that describes the observed data.

This approach is not limited to neuroscience or psychology and could be adapted for use in various fields of research where Bayesian linear regression is or could be used.

The project is implemented using the programming language `Python` and mainly its packages `equation_tree`, `pytorch`, `pandas`, `numpy`, `seaborn`,
`matplotlib` and `pyro`.

## Bayesian Linear Regression

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blr.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Bayesian linear regression model with the dependent variable $y_i$ sampled from a normal distribution where the mean is the point on the regression line and $\sigma^2$ is the variance. In this case, the prior distributions for the parameters are assumed to be normally distributed.
</div>

Bayesian and frequentist statistics are the two main branches of statistics employed in science. In frequentist statistics, probabilities are seen as a generalization of the facts of random events in repeated execution {% cite bayesian_data %}. On the other hand,

> "[i]n Bayesian statistics, probability is used as the fundamental measure or yardstick of uncertainty"{% cite bayesian_data %}

Put another way, in Bayesian statistics, probabilities can be thought of as the degree of belief or support of a rational agent who is uncertain about the outcome of a random event {% cite degree_of_belief %}.

The ability to explicitly model uncertainty is considered one of the main advantages of Bayesian statistics over frequentist statistics {% cite bayesian_data bayesianIncreasing %}. In addition, this approach takes advantage of prior knowledge about the world to obtain valuable results with a limited amount of data. While frequentist methods are still more common in psychological research, the rate of Bayesian methods is increasing {% cite pychology_bayes %}.

Although there are many different methods in Bayesian statistics, the focus here is on Bayesian linear regression, which is briefly described below.

In Bayesian linear regression, the goal is to fit a regression line $\hat{Y}$ described by $$\hat{Y} = \beta_{0} + \beta_{1} X$$ to the observed data. Here, $\beta_{0}$ refers to the intercept with the $y$-axis and $\beta_{1}$ refers to the slope of the regression line.
From that the formula for a value $y_i$ can be derived:
$$\hat{Y}_{i} = \beta_{0} + \beta_{1} X_{i} + \epsilon_{i}$$

In this case, $\epsilon_i$ is the individual error of every sample $i$ from the regression line. The goal is to find the posterior distribution for the parameters $\beta_0$ and $\beta_1$, rather than a point estimate.

Bayesian hierarchical regression is an approach to Bayesian modeling that can be used when the assumption of independence is not satisfied in the dataset. This can be the case when each participant in an experiment contributes multiple data points. The absolute numbers between participants may be very different while the relationship between the dependent and independent variables may still be the same. Bayesian hierarchical regression can address this problem by modeling each participant at one level and the group at another level.

## Theoretical Framework

This section will go into detail about how the mentioned expansion to nonlinearity in Bayesian regression is implemented and what the requirements are for it to work. Effectively, the nonlinearity is introduced by replacing the parameters $\beta_0$ and $\beta_1$, which are the intercept and slope from the linear regression model, with the $\beta_0$ to $\beta_n$, which are the $n$ latent units from the latent space of the autoencoder.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/stat_model.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The proposed model consists of $\mu$, $\sigma^2$, $x_i$, $y_i$, just like the simple linear model. However, the parameters $\beta_0$ and $\beta_1$ are extended to $n$ parameters where $\beta_0, \beta_1, ..., \beta_n$ are the units of the latent space in the autoencoder which have a normal distribution as prior. $y_i$ is then calculated by inserting $x_i$ into the equation derived from the decoder $f_\theta$ when given the latent space units $\beta_0, \beta_1, ..., \beta_n$ as input.
</div>

As mentioned, this work creates a neural network structure that finds a latent space representation that represents semantically similar equations by their distance in the latent space representation.
However, to achieve this, there needs to be a definition of semantic similarity. In linguistics, semantics refers to the meaning that a word provides while syntax refers to the structure of the word {% cite chomsky %}. In the context of mathematical equations here this idea is borrowed from linguistics which means that semantics refers to the structure of the function graph or the mathematical meaning of the symbols while syntax refers to the structure and building blocks of the function term.
In this particular case, semantic similarity is a measure of distance with the values of the function graph being more similar the closer they get. For example, $f(x) = cos(x)$ and $g(x) = sin(x+ \frac{\pi}{2} )$ should have a distance of zero in a perfect latent space representation as they are semantically equal in the mathematical sense because $cos(x) = sin(x+ \frac{\pi}{2})$.
There are different ways to measure semantic similarity between two equations.
Within the scope of this contribution, I will focus on the meaning of the area between the function graphs which implies that the smaller the area, the higher the similarity.
This means that the similarity of any two functions is anti-proportional to the integral of the subtraction of the two functions. \footnote{For the sake of simplicity and because it is not relevant in the practical implementation, the two functions $f$ and $g$ are considered to have no interception with each other.}

$$sim(f,g) \propto  \frac{1}{ \mid \int f(x) - g(x) dx \mid}  $$

In the following, the distance function

$$d(f,g) = \left| \int f(x) - g(x) dx \right| $$

will be used to measure how different two functions $f$ and $g$ are. This difference is here defined to be antiproportional to the similarity.

$$sim(f,g)  \propto \frac{1}{d(f,g)}$$

Therefore, if the distance is minimal, then the similarity is maximal.

In practice, this can be approximated using the Manhattan distance of the value vectors times the length of the interval $l$ and divided by the number of data points.

$$d(f,g) \approx \left( \sum_i^n \left| f(x_i) - g(x_i)\right| \right) \frac{l}{n} $$

Syntax is the second concept that is borrowed from linguistics. Syntactically similar refers to the similarity of the written equation. This means that $2x$ and $-2x$ would be considered syntactically equal as the only syntactic difference is the constant, however, semantically, they are very different since the area between the curves is relatively large, with four in the interval $[-1, 1]$. In this case, two equations with different constants are defined to be syntactically equivalent because the equation is considered in the form where the constant is replaced by a placeholder.

# Methods

## Dataset

The dataset is generated using the `equation-tree` package and every equation in the dataset consists of three parts:

- Function Term
- Constant
- Values of the Function Graph

Hence, there is a two-dimensional tensor for the function terms with the shape of `(batch size, maximum term length)`, a two-dimensional tensor for the constants with the shape `(batch size, 1)`, and a three-dimensional tensor for the $x$ and $y$ values of the function graph with the shape `(batch size, 2, 50)`.

The dataset contains 6000 equations and is split into a train and a test set. The training dataset consists of 80% of the total data and the test dataset contains the leftover data. All the evaluations shown in the result section are created by using the test data, which is not used for training.

### Function Term in Prefix Notation

Especially for the autoencoder with n-hot encoded equation symbols, the function terms must be represented as a sequence since the goal is to predict a sequence of mathematical symbols that make up the new function term. Therefore, the prefix notation, also known as the Polish notation, was chosen. This notation converts the tree representation of a function term, where the nodes are composed of elementary functions, operators, variables, and constants, into a sequence.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/eq_tree.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Equation tree for the function term $(1-2) \cdot 3$ which can be transformed to the prefix notation × - 1 2 3.
</div>

The notation is called prefix because the operator comes before the operands. Compared to the infix notation which is the standard representation of a function term, it does not need parentheses because it is unambiguous. For example, the function term $(1 - 2) \cdot 3$ in infix notation can be converted to the prefix notation `× - 1 2 3`.

The difference between elementary functions and operators is that an elementary function defines how an equation element $f(x)$ is transformed while an operator defines how two equation elements interact.
The function complexity and therefore the number of unique elementary functions is kept simple to avoid redundancy and to enhance learning. It can, however, be easily increased in future applications.

The elementary functions used to create the dataset are the trigonometric function $\sin\left(f(x)\right)$ and the exponential function $e^{f(x)}$.
As far as the operators are concerned, there are operations like addition, subtraction, and multiplication.

For the training, the equations are generated randomly up to a certain tree depth. In the current scenario, the tree depth was chosen to be two. Another restriction to the equations is that there are only equations with a maximum of one dependent variable chosen.

The equations and their values are generated by using the \lstinline{equation-tree} python package. The model requires a fixed input shape and therefore the maximum length of a function term in prefix notation is calculated and all other function terms are filled with padding elements to reach the same length. If the maximum length of the function term is six, then padding would have to be added to the function term from above in the following form:

`× - 1 2 3 <PAD>`

The padding is embedded in the model like any other character.

### Constants

The constants are separated from the function terms since each constant is a floating point value that is predicted by the model and not a finite set of classes such as the other elements of the mathematical function. Therefore, the prefix function term contains a placeholder for the constant and not its real value. The real value, however, is needed to instantiate the equation and obtain the function graph with its values. Therefore, the real values of the constants are saved separately.
The number of constants per equation is set to exactly one to keep a consistent shape of the tensor without any padding.

Every equation is 100 times instantiated with different constants. So that there are multiple syntactically equal equations with different constants. This process enables the model to learn how different values for the constants change the function values.

### Function Values

The function values are calculated by evaluating every equation 50 times in an equally spaced interval between -1 and 1. These values are later used to calculate the semantic similarity of the equations. Another important aspect here is that there is no gap in the definition, as there would be when dividing by zero, for the equations in the evaluated interval. This is important to avoid complications when calculating the semantic similarity between two equations and because the goal is to derive continuous functions from the data.

## Model Architecture

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ba_model_classify.drawio.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The autoencoder architecture has two separate encoder networks for the function terms $x_t$ and the constants $x_c$ which are then concatenated and again encoded to the latent space representation $z$. The latent space representation $z$ is then decoded by another dense neural network before it is split into different decoder networks for the reconstructed function terms $\hat{x_t}$ and the reconstructed constants $\hat{x_c}$
</div>

There are two types of models used in this approach. Both implement a neural network architecture called an autoencoder, but the second model is a variation of this architecture that implements variational inference and is called a variational autoencoder.

An autoencoder is a neural network structure whose goal is to reconstruct the input data.
Therefore, it consists of an encoder structure and a decoder structure. The encoder propagates the input through the network into a, in most cases, lower dimensional space in which the input is embedded. Throughout this thesis, this lower-dimensional space will be referred to as the embedding or the latent space.
After encoding the input information into the latent space, the next step is to reconstruct the original representation from the latent space, and this task is achieved by the decoder network. The main advantage of this neural network architecture is that it performs a dimension reduction of the input which can be beneficial in many different applications such as image compression and denoising {% cite lucas_theis_lossy_2017 gondara_medical_2016 %}.

As illustrated, the encoder $h_\phi(x)$ is comprised of separate encoder networks for the function terms and the constants which are then combined in another encoder network that results in the embedding vector $z$. The dimension of $z$ is variable and the performance of different dimensions is tested in the results section.
It is important to note that the function values do not serve as input to the network and are merely used in the loss function to calculate the latent correlation loss.
The function term encoder consists of an embedding layer with the size of the vocabulary $vocab$ which is the number of unique symbols used. This embedding layer is then fed into a linear layer with 64 units and rectified linear units (ReLU) activation where $ReLU(x) = max(0,x)$.
It outputs a predefined dimension which is eight in this case. Lastly, the input is flattened so that the output shape is $(8 \cdot vocab)$. The encoder for the constants is simply a linear layer that expands the dimension of the constants from one to eight. Afterward, the output of the function term encoder and the output of the constant encoder are concatenated and fed to another encoder network where the input is passed through a two-layer multilayer perceptron of size 56 and 64 with ReLU activation after the first layer which outputs the latent space as a vector with its predefined dimension.

The decoder $f_\theta(z)$ then increases the dimension with two linear layers of size 64 and 56 and ReLU activation between them. The output is then split and reshaped to have the shape \lstinline{(batch_size, max_term_length, 8)} for the function terms and \lstinline{(batch_size, 8)} for the constants. The function term decoder then consists of one linear layer of size eight that outputs a tensor of shape \lstinline{(batch_size, max_term_length, vocab_size)} which are the logits for every possible symbol at any position in the function term. To construct the function term, the symbol with the highest value is chosen for every individual position in the function term.
Hence, the function term decoder reconstructs each character individually.
The decoder for the constants consists only of one linear layer that outputs one unit the value of which is the prediction for the constant.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/VAE.drawio.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    In the variational autoencoder there are two separate encoder and decoder networks for the function terms $x_t$ and the constants $x_c$ just like in the autoencoder. However, the combined encoder network $h_\theta(x)$ results in a mean vector $\mu$ and a vector for the log variance $\log \sigma^2$ from which the latent space vector $z$ is sampled and decoded in the combined decoder $f_\theta(z)$.
</div>

The variational autoencoder {% cite kingma_auto-encoding_2013 %} is a special kind of autoencoder that incorporates principles from variational inference instead of directly learning the latent representation. It is a probabilistic model which means that instead of mapping the input to a fixed vector, it is mapped to a multivariate distribution.

The implementation of this architecture is very similar to the one of the standard autoencoder with the main difference being in how the latent vector is derived. In this case, the encoder does not directly return $z$ but much rather a mean vector $\mu$ and a log variance vector $\log \sigma^2$. These vectors are then used to sample the latent space $z$. The reparametrization trick is applied because the randomness of the sampling operation prevents the direct application of backpropagation. The reparametrization trick separates the deterministic and stochastic parts of the sampling operation by taking a sample $\epsilon$ from a standard normal distribution and then computing $z$ as a deterministic function where $z = \mu + \sigma \cdot \epsilon$.

The variational autoencoder can be seen as an implementation of variational inference which means that it is attempting to find the posterior distribution of $z$ given the input $x$. For the implementation of a variational autoencoder, it is important to find the best network parameters $\theta$ or in other words, to find the maximum likelihood estimate for $\theta$ given the data points $x_1, \ldots, x_n \in \mathbb{R}^D$.

$$
\begin{equation}
    \operatorname{ELBO} = \mathbb{E}_{q_\phi(z|x_i)} [ \log p_\theta(x_i|z) ] - D_{KL}(q_\phi(z|x_i) \| p(z))
\end{equation}
$$

The detailed derivation of this formula can be found in the paper by {% cite kingma_auto-encoding_2013 %}.
The surrogate distribution and the optimal parameters can be found by maximizing $ELBO$ given $q$ and $\theta$.

$$
\begin{equation}
\hat{q, \theta}:=\arg \max _{q, \theta} \operatorname{ELBO}(q, \theta)
\end{equation}
$$

In the case of the variational autoencoder, the surrogate distribution is calculated by

$$
\begin{equation}
\begin{aligned}
q_\phi(\boldsymbol{z} \mid \boldsymbol{x}) & := \mathcal{N}\left(h_\phi^{(1)}(\boldsymbol{x}), \operatorname{diag}\left(\exp \left(h_\phi^{(2)}(\boldsymbol{x})\right)\right)\right) \\
& = \mathcal{N}\left(\mu, \operatorname{diag}\left(\exp \left(\log \sigma^2 \right)\right)\right)
\end{aligned}
\end{equation}
$$

where $h_\phi^{(1)}$ and $h_\phi^{(2)}$ are neural networks that map the input $x$ to a mean $\mu$ and a log-variance $\log \sigma^2$ to get the approximate posterior distribution.

It is not possible to directly calculate the ELBO because the calculation would involve integrating over the latent variables $z$. Thus, the ELBO is maximized by stochastic gradient ascent and this is where the reparametrization trick comes into play. The surrogate distribution $q_\phi(z \mid x)$ is therefore split into a deterministic function $g_\phi$ and a random variable $\epsilon \sim \mathcal{D}$ where $\mathcal{D}$ is a normal distribution $\mathcal{N}(0, \sigma)$ with the standard deviation derived from the logarithm of the variance and the deterministic function is calculated by $g_\phi(\epsilon, x) = \mu + \sigma \epsilon $. 
Thus, the function $g_\phi$ can be used to get $z$ as $z := g_\phi(\epsilon, x)$.

$$
\begin{equation}
    \operatorname{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x_i)} [ \log p_\theta(x_i|z) ]}_{reconstruction} - \underbrace{D_{KL}(q_\phi(z|x_i) \| p(z))}_{regularization}
\end{equation}
$$

Since both $\theta$ and $\phi$ are differentiable, it is now possible to calculate the gradient that is used for gradient descent.

In practice, the KL divergence term can be seen as a regularization term on the reconstruction loss which means that the model tries to reconstruct each $x_i$ while at the same time trying to achieve a standard normal distribution for the equations in the latent space $z$. Therefore, the KL divergence between the prior $p(z)$ and the posterior $q(z \mid x)$ is minimized.

For the results to be comparable, the specific variational autoencoder implemented has a very similar architecture compared to the described autoencoder. The function term encoder and the constant encoder have the same number and size of layers compared to the standard autoencoder. The encoder, however, consists of a different set of layers that encode the mean and the logarithm of the variance. These encoder networks both have the same number and size of layers as the standard autoencoder. The decoder network, as well as the function term decoder and the constant decoder, are the same as described for the standard autoencoder.

## Loss

This problem needs an adaptation of the basic reconstruction loss of an autoencoder. First of all, the reconstruction is split into two parts, the reconstruction of the function term syntax and the reconstruction of the constants. The function term reconstruction uses cross-entropy ($CE$) and for the measure of reconstruction of the constants, a mean squared error ($MSE$) is used.
Since the latent space should be semantically ordered, it needs to be regularized. Several approaches share this purpose, one of the most remarkable is contrastive loss {% cite contrastiveLoss %}. However, this method, like most of the methods to regularize the latent space, is used mainly for computer vision tasks. Hence, their application in this task is limited. Contrastive loss aims to regularize the latent space in a way that positive samples, meaning data points belonging to the same class, are close to each other in the latent space. However, this problem is a mixture of classification and regression. In that sense, there are potentially as many classes as there are equations since, for the similarity of the equations, both the function term and its constant are considered. Or, to put it in another way, there are only negative samples except for very rare cases where they are semantically equal but syntactically different which is very unlikely considering how the dataset is created. Moreover, the constants can take any real number between zero and ten. Therefore, contrastive learning approaches like contrastive loss {% cite contrastiveLoss %}, lifted structured loss {% cite song2015deep %}, and triplet loss {% cite TrippletLoss %} cannot be used in combination with this method to fulfill the goal of semantic ordering.
A second approach is rank-n-contrast {% cite zha2023rankncontrast %} and it uses data augmentation to create different values for the same label. While this might work in computer vision tasks, where an image of a dog is still an image of a dog even though minor changes are applied like flipping or cropping the image, it is impossible for the task at hand. The function values deterministically follow from the equation and there is no way of changing the equation values without changing the equation itself. Therefore, this thesis introduces a different loss function called latent correlation loss ($LC$) which is described in detail in the next section.

The individual components of the loss are added together to calculate the overall loss. The loss function of the standard autoencoder is therefore calculated by the formula

$$
\begin{equation}
    L_{AE} = CE(\hat{x_t}, x_t) + MSE(x_c, \hat{x_c}) + LC(v, z).
    \label{math:loss1}
\end{equation}
$$

Here, $x_t$ and $x_c$ are the original function terms and constants, while $\hat{x_t}$ and $\hat{x_c}$ are the reconstructed function terms and constants. $v$ are the function values and $z$ are the latent vectors.
The mean squared error between the original constants $x_c$ and the reconstructed constants $\hat{x_c}$ is calculated by the formula

$$
\begin{equation}
\operatorname{MSE}(x_c, \hat{x_c})=\frac{1}{n} \sum_{i=1}^n\left(x_{c,i}-\hat{x}_{c,i}\right)^2 .
\end{equation}
$$

where $n$ is the batch size.

The loss for the function term reconstruction is calculated by cross-entropy loss because it is a state-of-the-art loss function for multi-class classification tasks {% cite Demirkaya2020ExploringTR %}. It is calculated using the formula

$$
\begin{equation}
CE(\hat{x_t}, x_t) = \frac{\sum_{n=1}^N l_n}{N}, \quad l_n=-\sum_{c=1}^C w_c \log \frac{\exp \left(\hat{x}_{t, n, c}\right)}{\sum_{i=1}^C \exp \left(\hat{x}_{t, n, i}\right)} x_{t, n, c}
\label{ce_loss}
\end{equation}
$$

where $C$ is the number of classes, in this case, the number of unique symbols. $w$ is the weight and $l_n$ is the loss for an individual entry of the batch.
This is with a mean reduction where the overall $CE$ loss is the average batch loss.

For the variational autoencoder, there needs to be the KL divergence loss as an additional term with a weighting term $w$. The complete calculation for the loss of the variational autoencoder is therefore given by $L_{VAE}$

$$
\begin{equation}
    L_{VAE} = CE(\hat{x_t}, x_t) + MSE(x_c, \hat{x_c}) + LC(v, z) +w KL(\mu, \log(\sigma^2))
    \label{math:loss_vae}
\end{equation}
$$

The KL divergence loss is based on the KL divergence between the prior and the surrogate distribution.

$$
\begin{equation}
\begin{aligned}
KL\left(\mu, \log\left(\sigma^2\right)\right) & := D_{KL}\left(q_\phi\left(z|x\right) \| p\left(z\right)\right) \\
&  = D_{KL}\left(\mathcal{N}\left(\mu, \operatorname{diag}\left(\exp \left(\log \sigma^2 \right)\right)\right)\| p\left(z\right)\right)
\end{aligned}
\end{equation}
$$

### Latent Correlation Loss

The latent correlation loss aims to quantify how well the latent space represents the semantic meaning of the equations.

For the sake of simplicity, the loss measures distance, which is defined as the inverse similarity, instead of the similarity. The distance matrix of the equations captures the distance of every equation with every other equation defined by the distance of the function values which approximates the area between the two function curves.

$$
\begin{equation}
    d(f_1,f_2) = \left(\sum_i^n \left| f_1(x_i) - f_2(x_i) \right| \right) \frac{l}{n}
\end{equation}
$$

The distance matrix of the function values $F$ for $n$ equations is therefore defined by

$$
F=\begin{bmatrix}
d_{11}&d_{12}&d_{13}&\dots &d_{1n}
\\d_{21}&d_{22}&d_{23}&\dots &d_{2n}
\\\vdots &\vdots &\vdots &\ddots &\vdots &
\\d_{n1}&d_{n2}&d_{n3}&\dots &d_{nn} \\\end{bmatrix}
$$

The distance of two latent space vectors $w$ and $u$ in a $m$-dimensional latent space can simply be calculated by using the Euclidean distance

$$d(\vec{w}, \vec{u}) = \Vert \vec{w} - \vec{u} \Vert = \sqrt{\sum_{i=1...m} (w_{i} - u_{i})^2}$$

of the latent vectors. The distance matrix $L$ of the equation distance in the latent space holds the information for the latent space distance of every equation with every other equation.

$$
L=\begin{bmatrix}
d_{11}&d_{12}&d_{13}&\dots &d_{1n}
\\d_{21}&d_{22}&d_{23}&\dots &d_{2n}
\\\vdots &\vdots &\vdots &\ddots &\vdots &
\\d_{n1}&d_{n2}&d_{n3}&\dots &d_{nn} \\\end{bmatrix}
$$

These two matrices of dimension $n\times n$ would ideally correlate with one. This means that ideally for every change in one matrix, there is a change in the other matrix accordingly. This would mean that the semantic meaning is perfectly represented in the latent space.

To calculate the correlation, the distance matrix of the function values and the distance matrix of the latent space are flattened into 1D vectors which means that the dimension of $F$ and $L$ changes from $n\times n$ to $n^2$.

The correlation is calculated by the Pearson correlation

$$
r_{F, L}=\frac{\sum_{i=1}^n\left(F_{i}-\bar{F}\right)\left(L_{i}-\bar{L}\right)}{\sqrt{\sum_{i=1}^n\left(F_{i}-\bar{F}\right)^2 \sum_{i=1}^n\left(L_{i}-\bar{L}\right)^2}}
$$

where $\bar{F}$ and $\bar{L}$ are the sample mean of $F$ and $L$.

This correlation coefficient is then adapted to fit the properties of a loss function. The correlation coefficient can take numbers between zero and one. The latent correlation loss is therefore calculated by

$$
\begin{equation}
    LC = - r_{F,L} + 1
\end{equation}
$$

This formula is chosen because it reverses the relationship of the correlation, meaning that the higher the correlation, the smaller the loss. Furthermore, it makes sure that the loss cannot be negative.

## Training

The chosen optimizer is the stochastic gradient descent algorithm adam {% cite kingma2017adam %} which is an adaptive learning rate optimization algorithm that is commonly used for the training of deep learning models. The initial learning rate is chosen to be 0.01 but it is adapted by the optimizer to achieve better performance. The other default parameters are a weight of 0.0001 for the KL divergence and a latent dimension of four. Different values for some of these parameters are tested to find the optimal parameter values. Each parametric change of the models is evaluated ten times to get more expressive results as for some metrics there was a lot of variance between different trials.

## Evaluation Metrics

There are several important metrics to assess the performance of the model. They all measure one of the predefined properties. First, there is the function term reconstruction accuracy, which calculates how many of the function terms are correctly recovered. Secondly, there is the mean squared error between the original constants and the recovered constants.
These two metrics test the reconstruction ability to ensure that the latent space embedding of the model is not ambiguous and that one embedding encodes for exactly one function term and constant. Furthermore, it assesses if the decoder can reconstruct the original function term from the input. This also assesses the property of searchability as the space is only searchable if the equations are properly reconstructed otherwise there is no coherence between the semantic ordering that is calculated based on the input during training and the reconstructed equations.

Another metric is the correlation coefficient between the distance matrix of the latent space and the distance matrix of the area between the function curves. These two distance matrices should correlate close to one because this would indicate that the semantically similar equations are close to each other in the latent space which is one of the main goals of the embedding. This correlation coefficient is calculated in the same way.

To fulfill the property of continuity, each point in the latent space must evaluate to a valid equation. The design of the model allows for impossible equations to occur. This can, for example, be the case when the model predicts padding in the middle of the function term like `['*', '+', 'x', '<PAD>', 'c_1']`, would be converted to

$$f(x) = (x+<PAD>)*c_1$$

which is an equation that cannot be interpreted as the padding has no mathematical value or meaning. There are many more ways that the model can produce mathematically impossible equations. This can be assessed by generating several random embedding vectors and evaluating what percentage of them result in valid equations when decoded by the model.

# Results

## Performance per Number of Latent Dimensions

The figures show how changing the latent dimensions affects performance. This means that only the size of the latent vector $z$ is changed, and everything else about the model remains the same.
The standard autoencoder achieves up to 100% accuracy in function term reconstruction for latent dimensions as small as two.
The constant MSE in the standard autoencoder improves up to about four dimensions and then stays between 0.0005 and 0.0010.
The mean correlation coefficient of the standard autoencoder improves up to about four dimensions and then stays above 0.99 from then on.

The variational autoencoder shows a mean reconstruction accuracy of 99% and higher from just two dimensions on as well.
The average constant MSE ranges form 0.007 to 0.008 for latent dimensions of two and higher. It only slightly increases for very large latent dimensions of 64 and 128 where it takes values of 0.013 and 0.015.
The average latent correlation of the variational autoencoder increases with the number of latent dimension up until four and five dimensions where it has the highest values with 97% accuracy. Other than the standard autoencoder, it starts to decrease again from then on which is very visible for the dimension of 128 where the average correlation drops to 89%.

## Comparison of Standard and Variational Autoencoder

The standard autoencoder performs slightly better than the variational autoencoder in the correlation between latent and value distance with an average of 0.95 across all dimensions compared to 0.92 in the variational autoencoder. The standard autoencoder is also better in terms of constant MSE, where it achieves an average MSE of 0.025 compared to 0.038 for the variational autoencoder. With regard to the average function term reconstruction accuracy, they have a comparable performance with both times 92%.

Therefore, as for the model with n-hot encoded equation elements, the standard autoencoder performs slightly better regarding the evaluation metrics.  
Nevertheless, the latent space of the variational autoencoder is denser and more regularized.

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/latent_space_ae_classify.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vae_latent_space_classify.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This graph shows the first two dimensions of a four-dimensional latent space for both, the standard autoencoder (left) and the variational autoencoder (right). Each point is the embedding of an encoded equation. The Variational autoencoder results in a more dense embedding.
</div>

## Weighting Factor for Kullback Leibner Divergence

This section examines how the weighting for the KL divergence affects the performance and the other losses. The KL divergence loss is required for the regularization of the latent space to be close to a distribution, which in this case is a standard normal distribution.
The performance of the variational autoencoder depends strongly on the weight assigned to the Kullback-Leibner divergence.
The correlation between the latent and the value distance decreases strongly with a KL divergence weight greater than 0.0001.
The accuracy of the function term reconstruction shows a strong decrease between a weight of 0.001 and 0.01, with an extremely low accuracy of less than 8% for weights greater than 0.001.
Coherently, the constant MSE rises strongly for weights greater than 0.01.
A weight of zero has the best performance across all evaluation metrics.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/kl_weights.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of different weighting factors for the KL divergence loss. The latent correlation decreases with an increase in weighting. The meaning of the box plots is as described in figure caption.
</div>

The KL divergence has a negative effect on the other losses. This is illustrated in the figure above, where it can be seen that the constant loss, the reconstruction loss as well as the latent correlation loss increase with an increase in the KL divergence weight.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/kl_losses.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The different loss functions (left) and the KL divergence loss (right) for different weighting factors of the Kullback Leibner divergence loss.
</div>

Even a weighting as low as $10^{-5}$ results in a notable reduction in the KL loss relative to the KL loss when the KL divergence is not considered. Values of 0.0001 and below offer a substantial decline in the KL loss without compromising the performance of the evaluation metrics.

## Latent Correlation Weighting

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/correlation_weighting.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Performance with different weighting factors for the latent correlation loss in the standard autoencoder (AE) and the variational autoencoder (VAE). The interpretation of the box plot is as  previously described.
</div>

This section examines the impact of varying weighting terms for the latent correlation loss on the performance of evaluation metrics. The results are illustrated in the figure above.
It can be observed that as the weight for the latent correlation loss in the standard autoencoder increases, the correlation between value distance and latent distance also increases, reaching a correlation of 1.0.
The function term reconstruction accuracy is, with an average accuracy higher than 97%, high across all values for the weighting.
Nevertheless, the constant MSE exhibits a pronounced increase with weightings exceeding ten.

The function term reconstruction accuracy in the variational autoencoder is high across all values for the latent correlation weighting, with only a slight decrease observed at a weighting of 1000.
Additionally, in the case of the variational autoencoder, the latent correlation increases with an increase in the weighting term.
Similarly, the constant MSE increases with the weighting, though not to the same extent as in the standard autoencoder.
The variational autoencoder still achieves a more than three times lower MSE of 0.03 compared to the standard autoencoder at a latent correlation weighting of 1000.
A higher weighting for the latent correlation loss increases the correlation without a strong decrease in the reconstruction accuracy. This way, the model can achieve a correlation of up to 0.99.

## Learning Rate

Although the learning rate is optimized during training, the initial learning rate appears to be crucial and is therefore examined in this section.
The function term reconstruction accuracy is high for the learning rates of 0.0001, 0.001, and 0.01 in both the standard autoencoder and the variational autoencoder. Moreover, performance decreases significantly for learning rates smaller or larger than these values.
For the constant MSE, the learning rate of 0.1 with an MSE of 50 on average for the variational autoencoder and an MSE of 10 for the standard autoencoder results in poor performance. However, other values for the learning rate result in low values for the constant MSE.
The average latent correlation is the highest for a learning rate of 0.001 in both the standard and the variational autoencoder, with correlations of 0.990 and 0.986, respectively. It is observed that the correlation decreases for learning rates higher or lower than 0.001.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/learning_rate_result.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of the performance for different values for the initial learning rate in the standard autoencoder (AE) and the variational autoencoder (VAE). The box plot can be interpreted as previously described.
</div>

## MCMC sampling on the Latent Space

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MCMC_vae_6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MCMC_vae_dist_6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example of a case where the MCMC algorithm derived the right function term and almost the exact constant. On the left are the sampled and the original equations shown and on the right is the distribution for each unit of the eight-dimensional latent space.
</div>

As previously stated, the latent space is designed to sample equations from it. This hypothesis is validated by creating a Markov chain Monte Carlo (MCMC) model with the `pyro` library, which samples from this space to determine the feasibility of the concept in principle. Indeed, the present findings confirm that the MCMC algorithm can be run on the embedding and that the distribution over the latent variables can be identified. In certain instances, the algorithm is able to identify the correct equation by utilizing the embedding generated by the variational autoencoder, as illustrated in the figure above. Moreover, the algorithm is capable of identifying the distribution over the latent dimension.

However, in other instances, the algorithm remains within a local optimum and provides a sub-optimal solution. This is shown in the figure below where the derived equations are similar but not equal to the original equation.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MCMC_vae_11.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example of a sub-optimal solution for MCMC algorithm. It shows the graph of the real function $5x$ and the graphs of the resulting functions $cx-sin(x)$ and $cx-e^x$.
</div>

# Discussion

The autoencoder with one-hot encoded function terms, other than the autoencoder with n-hot encoded equation elements, creates a space that has all the needed properties. It is continuous as valid equations are guaranteed no matter the value of the latent space vector. It can achieve a high semantic ordering in the latent space, which is shown by the high correlation between latent and value distance. Furthermore, the results provide a proof of concept that shows that the latent space is searchable.

For both, the standard and variational autoencoders, latent dimensions as small as four seem to be appropriate because the correlation between latent and value distance increases up to this dimension.

For the weighting of the KL loss, 0.0001 appears to be good because it strongly decreases the KL loss without sacrificing performance in the measured evaluation metrics.
For the standard autoencoder, a weighting for the latent correlation loss of 10 seems to be suitable because it improves the performance of the latent correlation without a decrease in the reconstruction accuracy.
For the variational autoencoder, there could even be a weighting of 100 used as it does not seem to have negative effects on the other evaluation metrics. However, since it does not further improve the correlation between latent and value distance, there seems to be no reason to use 100 over 10 for a four-dimensional latent space.
A learning rate of 0.001 seems to be the best option, for both the standard and variational autoencoders, of the different learning rates compared. Especially correlation between latent and value distance decreases with learning rates higher or lower than 0.001.

It is shown that MCMC sampling on the latent space is possible. However, further investigation is required to identify the circumstances under which the algorithm can identify the true equation. Nevertheless, in every case, the algorithm identified a solution, which demonstrates the feasibility of this approach.

The latent space created by the autoencoder with n-hot encoded equation elements did not fulfill the property of continuity, no matter the configurations tested. It created a sparse space where not every vector evaluates to a mathematically possible equation.
Therefore, the model architecture was modified to guarantee the mathematical validity of the equation by forcing the model to choose from all the predefined possible function terms.
This architectural adjustment ensures that the autoencoder fulfills the property of continuity by default. The model also performed well on the metric of a semantically ordered search space which was shown by a high correlation between function value distances and latent space distances. Furthermore, the space is searchable which was shown by performing MCMC sampling on the space and finding the correct, or close to correct, equation.
Variational autoencoders are better suited for generative tasks (\cite{generative_vae}) and this task here is generative as the model generates equations with new constants that it has not encountered during training. Therefore, the variational autoencoder might produce the more preferable search space. Indeed, the variational autoencoder produced a more dense latent space, however, the performance of the MCMC sampling on the latent spaces created by the different kinds of autoencoders is not compared in this thesis. Further testing needs to be done to reach a conclusion on the topic.

Unfortunately, because they employ different datasets and equation complexities, the outcomes of the autoencoder with n-hot encoded equation elements cannot be meaningfully compared to the outcomes of the autoencoder with one-hot encoded function terms. The first strategy also failed to meet the basic prerequisite of continuity. For these reasons, the results of the two methods used in this thesis are not compared here.

It should be noted that this approach has certain limitations that can be addressed in the future.
The number of unique function terms is quite small in the second method with a tree depth of two. However, the number of classes can easily be increased to allow for more complex function terms. After increasing the tree depth, it might also be worthwhile to increase the number of possible constants and later even the number of variables. Especially with an increased number of unique function terms, it might be advisable to increase the dataset size, which can be easily done since the equations are generated.

At this current state, it is important to mention that the model is not explicitly penalized for equation complexity. This means that a simpler equation is not necessarily preferred over a more complex equation with the same error. With increasing equation complexity it might be worthwhile to introduce a mechanism that penalizes complexity. This could be done by sampling on embeddings of models trained with different tree depths and comparing the error and complexity of the resulting equations.

Another limitation is the simple structure of the encoder and decoder which only uses dense layers. Further studies could explore this issue further by investigating how different layers, such as transformer layers, can improve performance. These could potentially take better account of the sequential structure of the equations than the linear layers implemented in this thesis.
It might also be worthwhile to examine the use of the stress function of multidimensional scaling (\cite{mds}) instead of the latent correlation loss and compare the performances.

Multidimensional scaling is a statistical method that is used to scale data points in a lower-dimensional space based on their distances. Its objective is therefore quite similar to the goal of this work. It takes the distance matrix and tries to find the optimal distance persevering map by performing eigenvalue decomposition. In classical MDS (\cite{mds}; \cite{mds_gower}), also known as principal coordinate analysis (PCoA), the squared distance matrix is used to minimize the strain.

$$
\begin{equation}
\operatorname{Strain}_D\left(z_1, z_2, \ldots, z_n\right)=\sqrt{\left(\frac{\sum_{i, j}\left(b_{i j}-z_i^T z_j\right)^2}{\sum_{i, j} b_{i j}^2}\right)}
\end{equation}
$$

Here $z$ is the coordinate in the latent space and $b_{ij}$ are the elements of the matrix $B$ that is the double centering matrix derived from the squared distance matrix and used to perform eigenvalue decomposition. Similarly, the stress function of metric multidimensional scaling could be used.

$$
\operatorname{Stress}_D\left(z_1, z_2, \ldots, z_n\right)=\sqrt{\sum_{i \neq j=1, \ldots, n}\left(d_{i j}-\left\|z_i-z_j\right\|\right)^2}
$$

In this case, $d_{ij}$ would be the distance of the function values while $\left\|z_i-z_j\right\|$ would be the distance in the latent space. It is a bit more restrictive than the latent correlation loss introduced in this thesis as it penalizes the two distances to have the same value instead of possibly a multiple.

It is also important to note that this thesis did not investigate the implementation of this method to Bayesian hierarchical regression. Future research should certainly further test how the MCMC sampling performs under different circumstances like, for example, noisy data.
In addition, it may be worthwhile to try variational inference as an alternative to MCMC sampling.

There are also other approaches that introduce nonlinearity to Bayesian regression. One of which is the Bayesian multivariate adaptive regression spline (MARS) {% cite denison_bayesian_1998 %} which builds on {% cite mars %}. This approach divides the data into intervals that are divided by cut points and performs linear regression between these cut points. Although this is nonlinear, it does not result in a continuous function. Therefore, if the underlying equation is nonlinear, then it can only be approximated by this method. This makes it hard to gain insight from the results of this method.
Other approaches expand the linear equation to a polynomial equation like $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n$ and use the same strategy as for Bayesian linear regression to fit the model {% cite bayesianPolynomial %}. However, this requires knowledge about the equation and is very limited concerning the possible outcomes of the equation. These methods are, in contrast to the method provided in this thesis, quite restricted concerning the nonlinearity and insight they can provide.

As mentioned in the introduction, there are many different symbolic regression algorithms. One popular symbolic regression algorithm is sparse identification of nonlinear dynamical systems (SINDy) developed by {% cite sindy %}. It is a data-driven approach that aims to identify governing equations for dynamical systems. The core idea behind SINDy is to find a sparse representation of the dynamics using a library of candidate functions. Other than the method introduced in this thesis, SINDy focuses on dynamical systems and modeling how state variables evolve over time. Furthermore, the amount of candidate functions in the library should be limited as the matrix may become ill-conditioned otherwise. Therefore, the selection of candidate functions in the library needs to be carefully chosen for the specific problem and needs knowledge about the domain. Additionally, it represents the derivative as a linear combination of the candidate functions and is therefore limited in complexity.
There are also Bayesian approaches to the SINDy algorithm {% cite bayesian_sindy %}, however, they have similar restrictions as mentioned for the original SINDy algorithm.

This work sets the foundation for a new approach to symbolic regression that can expand the possibilities of Bayesian hierarchical regression.
This may enable hierarchical models that encode individual differences in latent space distance and model architecture. It also allows researchers to question the linearity of the relationships between independent and dependent variables. In the future, it may be applied to many more fields and enable the modeling of increasingly complex problem sets far beyond computational neuroscience and psychology.

# Conclusion

Recent advances in computing power have enabled data-driven approaches that are transforming many fields. It has enabled the use of complex machine learning and, in particular, deep learning models that can provide a great leap forward for science. Symbolic regression is at the heart of these efforts, and therefore of great importance for discovering and understanding the underlying principles that govern the world and reality at large. It can bridge the gap between deep neural networks and human-understandable models.
It allows us to capture complex relationships in the data that are not limited to linearity, while at the same time providing the possibility of insight and understanding. This thesis has introduced a new method to this endeavor that offers some advantages over current symbolic regression algorithms. However, there is still work to be done to realize its full potential and to understand how it compares to other symbolic regression algorithms in terms of performance.
