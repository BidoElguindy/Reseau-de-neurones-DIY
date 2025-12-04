"""
Neural Network Library - DIY Implementation

A from-scratch implementation of neural network components using only NumPy.
This module provides core building blocks for constructing and training
deep learning models without relying on high-level frameworks.

Components:
-----------
Modules:
    - Linear: Fully connected layer
    - Conv2D: 2D Convolutional layer
    - MaxPool2D: Max pooling layer
    - Flatten: Tensor reshaping layer
    - Sequentiel: Sequential container for layers
    - AutoEncodeur: Autoencoder architecture

Activation Functions:
    - TanH: Hyperbolic tangent
    - Sigmoid: Logistic sigmoid
    - Softmax: Softmax for multi-class classification
    - LogSoftmax: Log-softmax for numerical stability

Loss Functions:
    - MSELoss: Mean Squared Error
    - CrossEntropyLoss: Cross-entropy loss
    - BCE: Binary Cross-Entropy
    - NLLLoss: Negative Log-Likelihood

Optimizers:
    - Optim: Optimizer wrapper
    - SGD: Stochastic Gradient Descent

Author: Georges Elguindy
License: MIT
"""

import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):  # On fait hériter MSELoss de Loss
    def forward(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.mean(diff * diff, axis=1)

    def backward(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / batch_size

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step

       pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

import numpy as np

class Linear(Module):
    def __init__(self, input_size, output_size):
        """
        Initializes a linear transformation module.
        output = input @ W + b

        """
        super().__init__()
        self._parameters = {
            "W": np.random.randn(input_size, output_size) * 0.1,  # Poids initialisés aléatoirement
            "b": np.zeros((1, output_size))  # Biais initialisé à 0
        }
        self._gradient = {
            "W": np.zeros((input_size, output_size)),
            "b": np.zeros((1, output_size))
        }
        self.input_cache = None 

    def forward(self, X) :
        """
        Performs the forward pass of the linear layer.
        """
        # Add assert for shape checking
        assert X.shape[1] == self._parameters["W"].shape[0], \
            f"Input feature size mismatch. Expected {self._parameters['W'].shape[0]}, got {X.shape[1]}"

        self.input_cache = X  # Sauvegarde de X pour la backpropagation
        return X @ self._parameters["W"] + self._parameters["b"]

    def backward_update_gradient(self, X_input, delta) : 
        """
        Computes and accumulates gradients for weights (W) and biases (b).
       
        #delta : Gradient of the loss with respect to the output of this layer,
                               shape (batch_size, output_size).
        """
        # Add assert for shape checking
        assert X_input.shape[0] == delta.shape[0], "Batch size mismatch between input and delta."
        assert delta.shape[1] == self._parameters["W"].shape[1], \
            f"Delta output feature size mismatch. Expected {self._parameters['W'].shape[1]}, got {delta.shape[1]}"


        self._gradient["W"] += X_input.T @ delta  # Gradient des poids
        self._gradient["b"] += np.sum(delta, axis=0, keepdims=True)  # Gradient du biais

    def backward_delta(self, X_input, delta): # Changed input to X_input
        """
        Computes the gradient of the loss to be propagated to the previous layer.
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        shape (batch_size, input_size).
        """
        # Add assert for shape checking
        assert delta.shape[1] == self._parameters["W"].shape[1], \
            f"Delta output feature size mismatch. Expected {self._parameters['W'].shape[1]}, got {delta.shape[1]}"

        return delta @ self._parameters["W"].T  # Propagation du gradient

    def update_parameters(self, gradient_step = 1e-3):
        """
        Updates the layer's parameters (W and b) using the accumulated gradients.
        """
        self._parameters["W"] -= gradient_step * self._gradient["W"]
        self._parameters["b"] -= gradient_step * self._gradient["b"]
        # self.zero_grad() # Moved zero_grad to Optim.step and SGD loop

    def zero_grad(self) :
        """
        Resets the accumulated gradients for W and b to zero.
        """
        for key in self._gradient:
            self._gradient[key].fill(0)

class TanH(Module):
    def __init__(self):
        super().__init__()  # Hérite correctement de Module

    def forward(self, X):
        self.output = np.tanh(X)  # Sauvegarde la sortie pour backward
        return self.output

    def backward_delta(self, input, delta):
        return delta * (1 - self.output ** 2)  # Dérivée de TanH

    def backward_update_gradient(self, input, delta):
        pass 

    def update_parameters(self, gradient_step):
        pass  
    def zero_grad(self):
        pass

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_clamped = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x_clamped))
        return self.out

    def backward_update_gradient(self, x, delta):
        # no parameters
        pass

    def backward_delta(self, x, delta):
        return delta * (self.out * (1 - self.out))

    def update_parameters(self, gradient_step):
        # No parameters to update for Sigmoid
        pass
    def zero_grad(self):
        pass

class Sequentiel(Module):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, X):
        self.inputs = [X]  # pour garder les entrées à chaque étape
        out = X
        for module in self.modules:
            out = module.forward(out)
            self.inputs.append(out)  # stocke la sortie pour backward
        return out

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, loss_fn: Loss) -> None:
            """
            Performs the backward pass through all modules in the sequence.
            """
            delta = loss_fn.backward(y_true, y_pred)
            for i in reversed(range(len(self.modules))):
                module = self.modules[i]
                # self.inputs[0] is the original X to the sequential model
                # self.inputs[1] is the output of modules[0] (and input to modules[1])
                # So, input to modules[i] is self.inputs[i]
                input_to_module_i = self.inputs[i]

                module.backward_update_gradient(input_to_module_i, delta)
                delta = module.backward_delta(input_to_module_i, delta)

    def update_parameters(self, learning_rate):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
    
class Optim:
    def __init__(self, net, loss_fn, eps):
        self.net = net
        self.loss_fn = loss_fn
        self.eps = eps

    def step(self, batch_x, batch_y):
        self.net.zero_grad()
        yhat = self.net.forward(batch_x)
        loss = self.loss_fn.forward(batch_y, yhat)
        self.net.backward(batch_y, yhat, self.loss_fn)
        self.net.update_parameters(self.eps)
        return np.mean(loss)  # utile si on veut suivre l'évolution de la perte

def SGD(optimizer, X_train, y_train, batch_size, epochs, verbose=True):
    N = X_train.shape[0]
    epoch_average_losses = []  

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        batch_losses_for_this_epoch = [] 
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            loss_of_current_batch = optimizer.step(batch_X, batch_y)
            batch_losses_for_this_epoch.append(loss_of_current_batch)

        avg_loss_this_epoch = np.mean(batch_losses_for_this_epoch)
        epoch_average_losses.append(avg_loss_this_epoch) 

        if verbose and (epoch % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs -1) : 
            print(f"Epoch {epoch} LR: {optimizer.eps if hasattr(optimizer, 'eps') else 'N/A'}, Epoch Loss: {avg_loss_this_epoch:.6f}")
            
    return epoch_average_losses 

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_shift = X - np.max(X, axis=1, keepdims=True)  # stabilité numérique
        exp_X = np.exp(X_shift)
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward_delta(self, input, delta):
        return delta  # pas besoin de recalculer si on utilise cross-entropy derrière

    def backward_update_gradient(self, input, delta):
        pass  # Pas de paramètres ,rien à faire

    def update_parameters(self, gradient_step):
        pass  # Pas de paramètres à mettre à jour
    def zero_grad(self):
        pass
class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        # Évite log(0) → ajoute un epsilon
        eps = 1e-15
        yhat_clipped = np.clip(yhat, eps, 1 - eps)

        # On applique la formule : -∑ y_i log(yhat_i)
        # Comme y est one-hot, on peut simplement faire : -log(yhat pour la bonne classe)
        loss = -np.sum(y * np.log(yhat_clipped), axis=1)  # un vecteur de taille batch
        return loss  # on peut faire .mean() à l’extérieur si besoin

    def backward(self, y, yhat):
        return yhat - y
    
class LogSoftmax(Module):
    def __init__(self):
        super().__init__()


    def forward(self, X):
        # stabilité numérique
        X_shift = X - np.max(X, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(X_shift), axis=1, keepdims=True))
        self.output = X_shift - log_sum_exp
        return self.output

    def backward_delta(self, input, delta):
        return delta  # pareil que Softmax si on combine avec NLLLoss

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step):
        pass
    def zero_grad(self):
        pass
class NLLLoss(Loss):
    def forward(self, y, log_yhat):
        loss = -np.sum(y * log_yhat, axis=1)
        return loss

    def backward(self, y, log_yhat):
        softmax = np.exp(log_yhat)
        return softmax - y
    
class BCE(Loss): 
    def __init__(self): 
        super().__init__()

    def forward(self, y_true, y_pred): 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_per_sample = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss_per_sample, axis=-1) # Ensure returns per-sample loss if input is multi-dimensional, or scalar if 1D

    def backward(self, y_true, y_pred): # Renamed parameters
     
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))
        # The division by y.shape[0] (batch_size) in original code is typical if the
        # overall loss is an average. If forward returns per-sample loss and Optim averages it,
        # then this gradient should not be pre-averaged here.
        # Let's assume Optim takes care of averaging the loss, so the gradient here
        # should be the derivative of the per-sample loss.
        return (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped) * y_true.shape[0])

class AutoEncodeur(Module):
    def __init__(self, encodeur, decodeur):
        super().__init__()
        self.encodeur = encodeur
        self.decodeur = decodeur

    def forward(self, x):
        self.encoded = self.encodeur.forward(x)
        self.decoded = self.decodeur.forward(self.encoded)
        return self.decoded

    def backward(self, X, X_hat, loss_fn):
        # Calcul du gradient de la loss par rapport à la sortie décodée
        delta = loss_fn.backward(X, X_hat)

        # Backward sur le décodeur avec X_hat = output reconstruit
        for i in reversed(range(len(self.decodeur.modules))):
            module = self.decodeur.modules[i]
            input_i = self.decodeur.inputs[i]
            module.backward_update_gradient(input_i, delta)
            delta = module.backward_delta(input_i, delta)

        # Backward sur l’encodeur avec le delta propagé du décodeur
        for i in reversed(range(len(self.encodeur.modules))):
            module = self.encodeur.modules[i]
            input_i = self.encodeur.inputs[i]
            module.backward_update_gradient(input_i, delta)
            delta = module.backward_delta(input_i, delta)

    def update_parameters(self, lr):
        self.encodeur.update_parameters(lr)
        self.decodeur.update_parameters(lr)

    def zero_grad(self):
        self.encodeur.zero_grad()
        self.decodeur.zero_grad()



class Conv2D(Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size # Can be an int (e.g., 3 for 3x3) or a tuple (kh, kw)
        self.stride = stride         # Can be an int or a tuple (sh, sw)
        self.padding = padding       # Can be an int or a tuple (ph, pw)

        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size

      
        limit = np.sqrt(6 / (input_channels * self.kh * self.kw + output_channels * self.kh * self.kw))
        self._parameters = {
            "W": np.random.uniform(-limit, limit, (output_channels, input_channels, self.kh, self.kw)) * 0.1, 
            "b": np.zeros((output_channels,)) 
        }
        self._gradient = {
            "W": np.zeros_like(self._parameters["W"]),
            "b": np.zeros_like(self._parameters["b"])
        }
        self.input_shape = None # To store input shape for backward pass
        self.X_col = None       # To store input as columns (im2col) for efficient convolution
                               

    def _pad_input(self, X):
        if self.padding == 0:
            return X
        # X shape: (batch_size, channels, height, width)
        if isinstance(self.padding, int):
            ph, pw = self.padding, self.padding
        else:
            ph, pw = self.padding
        
        return np.pad(X, ((0,0), (0,0), (ph,ph), (pw,pw)), mode='constant', constant_values=0)

    def _im2col(self, X_padded):
        pass

    def forward(self, X):
        # Input X shape: (batch_size, input_channels, input_height, input_width)
        self.input_shape = X.shape
        N, C_in, H_in, W_in = X.shape
        
        X_padded = self._pad_input(X)
        _, _, H_padded, W_padded = X_padded.shape

        # Stride can be int or tuple
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = self.stride if isinstance(self.stride, int) else self.stride[1]

        H_out = (H_padded - self.kh) // sh + 1
        W_out = (W_padded - self.kw) // sw + 1

        out = np.zeros((N, self.output_channels, H_out, W_out))

        # Store padded input for backward pass if using direct convolution logic
        self.X_padded_for_backward = X_padded 

        # Convolution loop (naive implementation)
        for n in range(N): # For each image in batch
            for c_out in range(self.output_channels): # For each output channel (filter)
                for h_out in range(H_out): # For each output height
                    for w_out in range(W_out): # For each output width
                        # Define the current receptive field in the padded input
                        h_start, w_start = h_out * sh, w_out * sw
                        h_end, w_end = h_start + self.kh, w_start + self.kw
                        
                        receptive_field = X_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        # Perform element-wise multiplication and sum
                        # Filter for this output channel: self._parameters["W"][c_out] (shape: C_in, kh, kw)
                        # Bias for this output channel: self._parameters["b"][c_out]
                        out[n, c_out, h_out, w_out] = np.sum(receptive_field * self._parameters["W"][c_out]) + \
                                                    self._parameters["b"][c_out]
        return out

    def backward_update_gradient(self, X_input_original, delta_output):
        # X_input_original: the input to this layer during forward pass (before padding)
        # delta_output shape: (N, C_out, H_out, W_out) - gradient from the next layer

        N, C_out, H_out, W_out = delta_output.shape
        _, C_in, H_in, W_in = self.input_shape 
        
        X_padded = self.X_padded_for_backward

        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = self.stride if isinstance(self.stride, int) else self.stride[1]

        # Gradient for weights (dW)
        # dW[c_out, c_in, kh, kw]
        # For each filter, iterate over its output map in delta_output
        # and correlate with the corresponding input patch
        for c_out in range(self.output_channels):
            for c_in in range(self.input_channels):
                for r in range(self.kh): # kernel row
                    for s in range(self.kw): # kernel col
                        # Sum over all N samples and all H_out, W_out positions
                        grad_sum = 0
                        for n in range(N):
                            for h_out in range(H_out):
                                for w_out in range(W_out):
                                    h_padded_start = h_out * sh + r
                                    w_padded_start = w_out * sw + s
                                    # Ensure within bounds of X_padded
                                    if h_padded_start < X_padded.shape[2] and w_padded_start < X_padded.shape[3]:
                                        grad_sum += X_padded[n, c_in, h_padded_start, w_padded_start] * \
                                                    delta_output[n, c_out, h_out, w_out]
                        self._gradient["W"][c_out, c_in, r, s] += grad_sum


        # Gradient for biases (db)
        # Sum delta_output over batch (N), height (H_out), and width (W_out) for each output channel
        self._gradient["b"] += np.sum(delta_output, axis=(0, 2, 3))


    def backward_delta(self, X_input_original, delta_output):
        # Calculate dL/dX_input (gradient to propagate to the previous layer)
        # This involves a "full convolution" of delta_output with rotated filters (W)
        # Output shape should be same as X_input_original: (N, C_in, H_in, W_in)

        N, C_out, H_out, W_out = delta_output.shape
        _, C_in_orig, H_in_orig, W_in_orig = self.input_shape # Original input shape
        
        # Create an empty gradient map for the input (padded dimensions initially)
        # (This is dL/dX_padded)
        dX_padded = np.zeros_like(self.X_padded_for_backward)

        # Stride
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = self.stride if isinstance(self.stride, int) else self.stride[1]
        
        filters_W = self._parameters["W"] # Shape: (C_out, C_in, kh, kw)

        # Iterate over each sample in the batch
        for n in range(N):
            # Iterate over each output channel of delta_output (which corresponds to a filter)
            for c_out in range(C_out):
                # Iterate over the spatial dimensions of delta_output
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # The region in dX_padded affected by this delta_output element
                        h_start, w_start = h_out * sh, w_out * sw
                        h_end, w_end = h_start + self.kh, w_start + self.kw
                        
                        # Get the current delta value
                        d_val = delta_output[n, c_out, h_out, w_out]
                        
                        # Add (filter_weights * d_val) to the corresponding region in dX_padded
                        # The filter weights are filters_W[c_out] (shape C_in, kh, kw)
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += filters_W[c_out, :, :, :] * d_val
        
        # Unpad dX_padded to get dL/dX_input
        if self.padding == 0:
            return dX_padded
        else:
            if isinstance(self.padding, int):
                ph, pw = self.padding, self.padding
            else:
                ph, pw = self.padding
            return dX_padded[:, :, ph:H_in_orig+ph, pw:W_in_orig+pw]


    def zero_grad(self):
        self._gradient["W"].fill(0)
        self._gradient["b"].fill(0)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["W"] -= gradient_step * self._gradient["W"]
        self._parameters["b"] -= gradient_step * self._gradient["b"]


# Flatten module to connect Conv layers to Linear layers
class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.input_shape_forward = None

    def forward(self, X):
        self.input_shape_forward = X.shape # (batch_size, channels, height, width)
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1) 

    def backward_delta(self, X_input_to_flatten_not_used, delta_from_linear):
        # delta_from_linear is 2D (batch_size, flattened_features)
        # Reshape it back to the original 4D input shape of this Flatten layer
        return delta_from_linear.reshape(self.input_shape_forward)

    # Parameter-less methods
    def backward_update_gradient(self, X_input, delta): pass
    def update_parameters(self, gradient_step): pass
    def zero_grad(self): pass


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size 
        
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size
            
        self.stride = stride if stride is not None else kernel_size 
        if isinstance(self.stride, int):
            self.sh, self.sw = self.stride, self.stride
        else:
            self.sh, self.sw = self.stride
            
        self.input_shape_forward = None
        self.max_indices = None # To store the indices of the max values

    def forward(self, X):
        # Input X shape: (batch_size, channels, input_height, input_width)
        self.input_shape_forward = X.shape
        N, C, H_in, W_in = X.shape

        H_out = (H_in - self.kh) // self.sh + 1
        W_out = (W_in - self.kw) // self.sw + 1

        out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
       
        self.max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int) # Stores (row, col) indices

        for n in range(N): 
            for c in range(C): 
                for h_idx_out in range(H_out): 
                    for w_idx_out in range(W_out): 
                        # Define the current pooling window in the input
                        h_start = h_idx_out * self.sh
                        h_end = h_start + self.kh
                        w_start = w_idx_out * self.sw
                        w_end = w_start + self.kw
                        
                        window = X[n, c, h_start:h_end, w_start:w_end]
                        
                        # Find the max value in the window
                        max_val = np.max(window)
                        out[n, c, h_idx_out, w_idx_out] = max_val
                        
                        # Find the (row, col) index of the max_val *within the window*
                        # np.unravel_index can find it in the flattened window, then convert back
                        # For simplicity, we find the first occurrence if multiple max values exist.
                        idx_in_window = np.unravel_index(np.argmax(window), window.shape)
                        
                        # Store the original input indices
                        self.max_indices[n, c, h_idx_out, w_idx_out, 0] = h_start + idx_in_window[0]
                        self.max_indices[n, c, h_idx_out, w_idx_out, 1] = w_start + idx_in_window[1]
        return out

    def backward_delta(self, X_input_not_used, delta_output):
        # delta_output shape: (N, C, H_out, W_out) - gradient from the next layer
        # We need to create dL/dX_input, which has the shape of the original input to forward
        N, C, H_out, W_out = delta_output.shape
        _, _, H_in, W_in = self.input_shape_forward
        
        dX_input = np.zeros(self.input_shape_forward, dtype=delta_output.dtype)

        for n in range(N):
            for c in range(C):
                for h_idx_out in range(H_out):
                    for w_idx_out in range(W_out):
                        # Get the (row, col) where the max came from in the original input
                        h_max_idx_in = self.max_indices[n, c, h_idx_out, w_idx_out, 0]
                        w_max_idx_in = self.max_indices[n, c, h_idx_out, w_idx_out, 1]
                        
                        # Add the gradient from delta_output to this specific location
                        # If multiple output elements derived their max from the same input element
                        # (can happen with overlapping pooling or stride < kernel_size),
                        # their gradients should sum up at that input location.
                        dX_input[n, c, h_max_idx_in, w_max_idx_in] += delta_output[n, c, h_idx_out, w_idx_out]
                        
        return dX_input

    # MaxPool2D has no learnable parameters
    def backward_update_gradient(self, X_input, delta):
        pass

    def update_parameters(self, gradient_step):
        pass # No parameters to update

    def zero_grad(self):
        pass # No gradients to zero for this layer itself