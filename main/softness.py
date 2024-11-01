import numpy as np
import re
import mdtraj as md
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def D_sq_min(traj, frame, delta_t, cutoff, lambda_s):
    """
    This function is for calculation of D_min to get the rearangement of the particles

    Input: 
        traj: mdtraj object that used for calculating
        frame:
            int
            which start frame to calculate
        delta_t: 
            constant  
            The time range of the displacement of a particle 
        cutoff:
            constant
            The cutoff radius to count the neighbors of the target particle in nano meters
        lambda_s:
            constant
            The stress tensor
    Ouput:
        D_sq_min:
            numerical
            The final calculation results of the D min
    """
    topology = traj.topology
    lithium_index = [ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ]
    phosphorus_index = [ atom.index for atom in topology.atoms if (re.search(r'P', atom.name) != None) ]
    sulfur_index = [ atom.index for atom in topology.atoms if (re.search(r'S', atom.name) != None) ]
    cholride_index = [ atom.index for atom in topology.atoms if (re.search(r'Cl', atom.name) != None) ]
    D_sq_min = []
    next_frame = int(delta_t + frame)
    
    neighbors_init = md.compute_neighborlist(traj[frame], cutoff) #[100:]
    neighbors_next = md.compute_neighborlist(traj[next_frame], cutoff) #[100:]
    for li_index in  lithium_index:
        pairs_init = [ [i, frame] for i in neighbors_init[li_index]]
        # note that to to tack the same neighbors the pairs_init is kept to the next frame
        pairs_next = [ [i, next_frame] for i in neighbors_next[li_index]]
        distances_init = md.compute_distances(traj[frame], np.array(pairs_init))
        distances_next = md.compute_distances(traj[next_frame], np.array(pairs_init))
        D_sq_min.append(np.mean((distances_next - lambda_s * distances_init)**2) )
    return D_sq_min

def Calc_radial_G(traj, frame, li_index, cutoff, mu_init, mu_inc, mu_final, L,):
    """
    This function calculates the Raidal distribution function of the target Lithium within the cutoff radius
    Input: 
        traj: mdtraj object
        frame:
            int
            start frame of the traj
        index:
            int
            Lithium index that need to be calculated
        cutoff:
            float
            cutoff radius of the shell that surrounding the lithium
        mu_init:
            float
            The start value for calculate the G_r function
        mu_inc:
            float
            The increment of the mu for calculating a series of the G_r
        mu_final:
            float
            The final value for the G_r calculation
        L:
            float
            thickness of the shell Default is cutoff/150
    """
    # calculte the R_ij
    neighbors= md.compute_neighbors(traj[frame], cutoff, [li_index], periodic=True) #[100:]
    pairs_init = [ [i, frame] for i in neighbors[0]]  
    R_ij = np.array(md.compute_distances(traj[frame], np.array(pairs_init))).T

    #Using R_ij to get the final G_r
    mu_one = np.array([ mu_now for mu_now in np.arange(mu_init, mu_final, mu_inc)])
    G_r = [ np.sum(np.e**(-(R_ij - mu  )**2/L**2)) for mu in mu_one]
    
    return G_r

def PrepareInput(traj, D_min, frame, ):
    """
    This function prepares the inputs for the softness ML-algorithm
    Input:
        traj: mdtraj object
        D_min: the non-square displacement calculated. 
            type: list
            shape: (n,)
        D_sq_min_0: the cutoff for spliting the soft and hard particles 
            type:float
            shape: 1
    """
    topology = traj.topology
    lithium_index = [ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ]
    X = [Calc_radial_G(traj=traj, frame=frame, li_index=index, cutoff=0.6, mu_init=0.3, mu_inc=0.1, mu_final=4, L=0.1)
         for index in lithium_index ]
    print(f"number of G_r functions {len(X[0])}")
    # get the label
    data = D_min
    bin_edges = np.arange(0, 0.6, 0.10)
    bin_indices = np.digitize(data, bins=bin_edges)
    bin_values = np.array(bin_edges)[bin_indices - 1]  # Subtract 1 to convert to 0-based indices
    y_new = bin_values/0.10
    y_new = y_new.astype(int)
    print("Done.")
    return X, y_new

def Softness(X, y, algorithm):
    """
    Softness classification
    """
    if algorithm == "logistic_regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, )
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define hyperparameters
        learning_rate = 0.001
        epochs = 20
        batch_size = 128
        output_units = np.unique((y)).shape[0]
        output_units = np.unique((y)).max()
        # Define logistic regression model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_units+1, activation='linear')
        ])
        
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        train_loss, train_accuracy = model.evaluate(X_train, y_train)
        print(f"Test accuracy: {test_accuracy}")
        print(f"Training accuracy: {train_accuracy}")
        return model
    elif algorithm == "svm":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, )
        svm = SVC(probability=True, kernel='rbf', random_state=1,  C=200,)
        svm.fit(X, y)
        print(f"Test accuracy: {svm.score(X_test,y_test)}")
        print(f"Training accuracy: {svm.score(X_train,y_train)}")
        return svm