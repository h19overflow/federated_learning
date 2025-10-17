This code defines the FlowerClient class, an individual participant in the
Federated Learning system for pneumonia detection. It uses the Flower
framework to manage client-side operations.
80
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
1. init method initializes a client with its unique ID, local data partitions
(training and validation from the RSNA sample), and image directory. It
computes client-specific class weights, instantiates a local LitResNet
model (ResNet50 V2 with custom head and specific configurations), and
sets up a client-specific XRayDataModule for data loading and
preprocessing. 2. get_parameters method retrieves the current model weights from the
client's LitResNet and returns them as NumPy arrays for transmission to
the server. 3. The set_parameters method updates the client's local model with
parameters received from the server (typically the global model). 4. fit method performs local training. It updates the local model with global
parameters, then trains it for a set number of epochs on its local data
using a PyTorch Lightning Trainer. It returns the updated local parameters,
the size of its training dataset, and an empty metrics dictionary. 5. evaluate method performs local evaluation. It updates the local model with
received parameters and evaluates it on its local validation data using a
PyTorch Lightning Trainer. It returns the local validation loss, the size of its
validation dataset, and a dictionary with local accuracy and loss. The FlowerClient class is crucial for the project's federated learning
simulation (Sections 3.4 and 3.5). It enables clients to collaboratively train the
ResNet50 V2 model on their private data without direct data sharing, supporting the development of a privacy-preserving pneumonia detection
system.
81
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
class FlowerClient(fl.client.NumPyClient):
def __init__(self, cid, train_df_client, val_df_client, image_dir):
self.cid = cid
self.train_df_client = train_df_client
self.val_df_client = val_df_client
self.image_dir = image_dir
self.class_weights_tensor_client =
compute_class_weights_for_pl(self.train_df_client, 'Target')
if self.class_weights_tensor_client is not None:
self.class_weights_tensor_client =
self.class_weights_tensor_client.to(DEVICE)
self.model = LitResNet(
base_model_weights=RESNET_WEIGHTS_CONFIG,
learning_rate=INITIAL_LR_CONFIG, weight_decay=WEIGHT_DECAY_CONFIG, dropout_rate=0.5,
fine_tune_layers_count=FINE_TUNE_OFFSET_CONFIG, class_weights_tensor=self.class_weights_tensor_client
)
self.model.to(DEVICE)
self.data_module = XRayDataModule(
train_df=self.train_df_client, val_df=self.val_df_client, # For client's local validation dataloader
image_dir=self.image_dir,
img_size=IMG_SIZE_CONFIG, batch_size=BATCH_SIZE_CONFIG, seed=SEED_CONFIG + int(cid), # Vary seed slightly for client dataloaders
color_mode='rgb', num_workers=CLIENT_NUM_WORKERS_CONFIG, use_custom_preprocessing_val=USE_CUSTOM_PREPROCESSING_VAL_CONFI
82
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
G, # For val transforms
use_imagenet_norm=USE_IMAGENET_NORM_CONFIG
)
self.data_module.setup(stage='fit') # Prepares train_dataloader and
val_dataloader
def get_parameters(self, config):
return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
def set_parameters(self, parameters):
params_dict = zip(self.model.state_dict().keys(), parameters)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
self.model.load_state_dict(state_dict, strict=True)
def fit(self, parameters, config):
self.set_parameters(parameters)
trainer = pl.Trainer(
max_epochs=FL_LOCAL_EPOCHS_PER_CLIENT_ROUND, accelerator="gpu" if DEVICE.type == "cuda" else "cpu", devices=1 if DEVICE.type == "cuda" else None, enable_checkpointing=False,
logger=False, callbacks=[], deterministic=True, enable_progress_bar=False
)
trainer.fit(self.model, datamodule=self.data_module) # Uses train_dataloader
from data_module
return self.get_parameters(config={}), len(self.data_module.train_dataset), {}
def evaluate(self, parameters, config):
self.set_parameters(parameters)
83
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
trainer = pl.Trainer(
accelerator="gpu" if DEVICE.type == "cuda" else "cpu", devices=1 if DEVICE.type == "cuda" else None, enable_checkpointing=False,
logger=False, callbacks=[], deterministic=True, enable_progress_bar=False
)
results = trainer.validate(self.model, datamodule=self.data_module, verbose=False) # Uses val_dataloader
loss = results[0].get('val_loss', float('inf'))
accuracy = results[0].get('val_acc', 0.0)
return loss, len(self.data_module.val_dataset), {"accuracy": accuracy, "val_loss": loss}
3.4.2 Partitioning
This Python code segment details the data partitioning strategy for a two- client federated learning simulation, ensuring consistency with the centralized
model's data setup for a fair comparison, as discussed in Section 3.4 of the
"Building a Federated Learning System for Pneumonia Detection" report. The process begins by using the same initial training dataset (full_train_df, which is identical to train_df from the centralized configuration) and the main
84
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
validation dataset (global_val_df, identical to val_df). The IMAGE_DIR also
remains consistent. Federated Data Partitioning Steps:
1. Inter-Client Data Split (50/50):
a. The full_train_df is divided equally between two simulated clients,
train_df_client1 and train_df_client2. b. This is achieved using train_test_split with the test_size=0.5 parameter, which allocates 50% of the full_train_df to train_df_client2 and the
remaining 50% to train_df_client1. c. To maintain class proportionality, this split is stratified based on the
'Target' column (stratify_col_train). The SEED_CONFIG ensures this
division is reproducible. 2. Intra-Client Local Train/Validation Split (75/25):
a. Each client's assigned portion of the data (train_df_client1 and
train_df_client2) is further subdivided into a local training set and a local
validation set. b. For Client 1, train_df_client1 is split into client1_actual_train_df (75% for
local training) and client1_local_val_df (25% for local validation). c. Similarly, for Client 2, train_df_client2 is split into client2_actual_train_df
(75%) and client2_local_val_df (25%). d. These intra-client splits are also stratified (using stratify_val1 and
stratify_val2) and utilize the SEED_CONFIG. The 25% test_size for these
local validation sets is consistent with the 25% proportion used for the
global_val_df in the centralized approach, ensuring uniformity in validation
data ratios.
85
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
This partitioning strategy ensures that the total data used for training across
all clients in the federated setup originates from the same dataset
(full_train_df) as the centralized model. The data is explicitly split 50/50
between the two clients, and the proportion of data held out for validation
within each client's local context (25%) mirrors that of the centralized model's
global validation set. The global_val_df is reserved for the final evaluation of
the aggregated federated model, enabling a direct and equitable performance
comparison against the centralized baseline. Stratification is applied at each
splitting stage to preserve class distributions.
full_train_df =train_df
global_val_df = val_df
IMAGE_DIR = IMAGE_DIR
stratify_col_train = full_train_df['Target'] if full_train_df['Target'].nunique() > 1 else
None
train_df_client1, train_df_client2 = train_test_split(
full_train_df,
test_size=0.5, random_state=SEED_CONFIG, stratify=stratify_col_train
)
stratify_val1 = train_df_client1['Target'] if train_df_client1['Target'].nunique() > 1 else
None
client1_actual_train_df, client1_local_val_df = train_test_split(
train_df_client1, test_size=0.25, random_state=SEED_CONFIG, stratify=stratify_val1)
stratify_val2 = train_df_client2['Target'] if train_df_client2['Target'].nunique() > 1 else
None
client2_actual_train_df, client2_local_val_df = train_test_split(
86
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
train_df_client2, test_size=0.25, random_state=SEED_CONFIG, stratify=stratify_val2)
3.4.3 Client Instantiation (client_fn)
The Flower federated learning framework requires a client_fn function. This
function acts as a factory, responsible for creating and returning a
FlowerClient instance whenever the simulation needs to interact with a
specific client.
In this project, client_fn is designed for a two-client simulation. It takes a client
identifier string (cid) as input (e.g., "0" or "1"). Based on this cid, the function retrieves the appropriate pre-partitioned
training and local validation DataFrames (client1_actual_train_df and
client1_local_val_df for cid="0", and client2_actual_train_df and
client2_local_val_df for cid="1"). These DataFrames are passed as copies
(.copy()) to the FlowerClient constructor, along with the common IMAGE_DIR,
to ensure data integrity across client instances. The FlowerClient constructor then initializes the client-specific LitResNet
model (with project-defined configurations for architecture, pre-trained
weights, learning rate, fine-tuning, and local class weights) and it's dedicated
XRayDataModule for handling its local data partition. This mechanism ensures that each simulated client is properly initialized with
its unique dataset and a configured local model, ready to participate in
federated rounds.
87
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
3.4.4 Server-Side Centralized Evaluation Configuration (get_evaluate_fn)
To monitor the performance of the globally aggregated model, a server-side
evaluation function is necessary. The get_evaluate_fn is a higher-order
function designed to provide this capability. get_evaluate_fn: takes the global hold-out validation DataFrame
(val_df_global) and its corresponding image directory (image_dir_global) as
input. This global validation set is kept separate and is not used by any client
during its local training phase, ensuring an unbiased assessment of the global
model's generalization.
It returns an inner function, evaluate_fn, which is then passed to the Flower
server's strategy (e.g., FedAvg) to be called at specified intervals (e.g., after
each aggregation round or at the end of the entire federated training process). The evaluate_fn itself performs the actual evaluation:
It receives the current federated round number (server_round) and the global
model parameters (parameters_ndarrays) from the server strategy.
It instantiates a fresh LitResNet model on the server-side. 1. It loads the received global model parameters into this evaluation model
and sets it to evaluation mode (model.eval()). 2. It sets up an XRayDataModule using the val_df_global to create a
DataLoader for inference. 3. The evaluation model then performs inference over the entire
val_df_global. The loss is calculated consistently using the model's _calculate_loss method, which can incorporate class weights specific to
val_df_global if necessary. 4. Key performance metrics (accuracy, precision, recall, F1-score, AUROC)
are computed using torchmetrics.
88
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
5. Crucially, for the final federated round, this function also generates and
saves a detailed classification_report and a confusion_matrix plot, providing a
comprehensive analysis of the final converged global model's performance on
the unseen global validation data. 6. The average loss and the dictionary of metrics are returned to the server
strategy, allowing for the tracking and logging of the global model's
performance progression throughout the federated learning process. These two functions, client_fn and get_evaluate_fn, are fundamental
components of the federated learning simulation setup, enabling the
controlled creation of distributed clients and the systematic evaluation of the
collaboratively trained global model. def client_fn(cid: str) -> FlowerClient:
return FlowerClient(cid, client1_actual_train_df.copy() if cid == "0" else
client2_actual_train_df.copy(), client1_local_val_df.copy() if cid == "0" else
client2_local_val_df.copy(),
IMAGE_DIR)
def get_evaluate_fn(val_df_global, image_dir_global):
def evaluate_fn(server_round, parameters_ndarrays, config):
class_weights = compute_class_weights_for_pl(val_df_global, 'Target')
if class_weights is not None:
class_weights = class_weights.to(DEVICE)
model = LitResNet(
base_model_weights=RESNET_WEIGHTS_CONFIG,
learning_rate=INITIAL_LR_CONFIG, weight_decay=WEIGHT_DECAY_CONFIG,
89
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
dropout_rate_head_1=0.5, dropout_rate_head_2=0.5,
fine_tune_layers_count=FINE_TUNE_OFFSET_CONFIG, class_weights_tensor=class_weights
)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in
zip(model.state_dict().keys(), parameters_ndarrays)})
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()
data_module = XRayDataModule(
train_df=pd.DataFrame(), val_df=val_df_global,
image_dir=image_dir_global,
img_size=IMG_SIZE_CONFIG, batch_size=BATCH_SIZE_CONFIG, seed=SEED_CONFIG, num_workers=GLOBAL_EVAL_NUM_WORKERS_CONFIG, use_custom_preprocessing_val=USE_CUSTOM_PREPROCESSING_VAL_CONFI
G, use_imagenet_norm=USE_IMAGENET_NORM_CONFIG
)
data_module.setup('validate')
val_loader = data_module.val_dataloader()
preds, labels, loss_sum, count = [], [], 0.0, 0
with torch.no_grad():
for x, y in val_loader:
x, y = x.to(DEVICE), y.to(DEVICE)
logits = model(x)
loss = model._calculate_loss(logits, y)
loss_sum += loss.item() * x.size(0)
count += x.size(0)
90
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
preds.extend(torch.sigmoid(logits).cpu().numpy().ravel())
labels.extend(y.cpu().numpy().ravel())
preds, labels = np.array(preds), np.array(labels).astype(int)
pred_classes = (preds > 0.5).astype(int)
avg_loss = loss_sum / count if count > 0 else float('inf')
metrics = {
'val_acc': torchmetrics.Accuracy(task="binary")(torch.tensor(pred_classes),
torch.tensor(labels)).item(),
'val_recall': torchmetrics.Recall(task="binary")(torch.tensor(pred_classes),
torch.tensor(labels)).item(),
'val_precision':
torchmetrics.Precision(task="binary")(torch.tensor(pred_classes),
torch.tensor(labels)).item(),
'val_auroc': torchmetrics.AUROC(task="binary")(torch.tensor(preds),
torch.tensor(labels)).item() if len(np.unique(labels)) > 1 else 0.0,
'val_f1': torchmetrics.F1Score(task="binary")(torch.tensor(pred_classes),
torch.tensor(labels)).item()
}
if server_round == FL_NUM_ROUNDS:
report = classification_report(labels, pred_classes, target_names=['Normal',
'Pneumonia'], zero_division=0)
with open(os.path.join(FL_RESULTS_DIR, "fl_classification_report_final.txt"), "w") as f:
f.write(f"Federated Round: {server_round}\n{report}\n\n")
for k, v in metrics.items():
f.write(f"{k}: {v}\n")
f.write(f"avg_loss: {avg_loss}\n")
cm = confusion_matrix(labels, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal',
'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.tight_layout()
91
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
plt.savefig(os.path.join(FL_RESULTS_DIR,
f'fl_confusion_matrix_round_{server_round}.png'))
plt.close()
return avg_loss, metrics
return evaluate_fn
3.5 Federated Averaging (FedAvg) Algorithm Implementation with
ResNet50 V2
With all the individual pieces in place – the clients ready with their local data
and models, and the server knowing how to evaluate the collective effort – it's
time to kick off the collaborative learning journey. This is where the
fl.simulation.start_simulation function from the Flower library steps in to
orchestrate the entire federated learning experiment. 3.5.1 The Federated Averaging (FedAvg) Formula
● is the number of participating clients
● is the updated global model at round t+1
● is the total number of samples across all clients
● is the local model weights from client k at round t
92
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
This formula ensures that clients with more data have a proportionally larger
influence on the global model. 3.5.2 Federated Simulation Execution (fl.simulation.start_simulation):
Think of fl.simulation.start_simulation as the command that brings our entire
federated system to life. It takes all our carefully prepared components and
instructions and runs the show. Here’s what we’re telling it:
1. Our Client Creator (client_fn=client_fn): We hand Flower our client_fn. This is like giving it a blueprint: "Whenever you need a client, use this
function to make one." Flower will use this to spawn our virtual hospitals
(clients), each with its unique slice of the RSNA dataset and its own
fresh copy of the ResNet50 V2 model. 2. Number of Participants (num_clients=2): We specify that our
federated network consists of two distinct clients. 3. Server's Rulebook
(config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS)): This
sets up the central server. The most important rule here is
num_rounds=FL_NUM_ROUNDS, which defines how many "rounds" of
learning our system will go through. Each round is a complete cycle of
learning and improvement. 4. The Learning Strategy (strategy=strategy): This is the heart of the
server's intelligence. We pass our pre-configured FedAvg strategy. This
strategy knows:
a. To involve all clients in training (fraction_fit=1.0). b. To rely on our special server-side evaluation (evaluate_fn) instead of
asking clients to evaluate themselves.
93
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
c. The initial set of "starter" model weights (initial_parameters) for the
very first global model. d. Crucially, it knows the FedAvg recipe for combining client
contributions. 5. Client Resources (client_resources=client_resources_config): We
give Flower an idea of the computing power each client has (e.g., if they
have a GPU or just CPUs). This helps Flower run the simulation
efficiently, especially if we were simulating many more clients. The process of a Federated Learning Round:
Once start_simulation is called, the federated learning process begins,
iterating through FL_NUM_ROUNDS. Each round is like a carefully
choreographed dance:
1. The Server's Invitation: The server, using the FedAvg strategy, decides it's time for a learning round. It selects the clients to participate
(in our case, both clients). 2. Dispatching the Global Model: The current global model's weights are
packaged up and sent from the server to each selected client. Imagine
the server emailing the latest blueprint to each hospital. 3. Local Learning at Each Client:
a. Each FlowerClient receives these global weights and updates its
local LitResNet model. b. Then, the real work begins: the client trains this model only on its
own private portion of the RSNA dataset for a set number of local
epochs (FL_LOCAL_EPOCHS_PER_CLIENT_ROUND). This is
where the model learns from that specific client's unique data
patterns.
94
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
4. Reporting Back: After their local training, clients don't send their data.
Instead, they send back their updated model weights – the result of their
local learning. They also report how many data samples they used, which is important for the server. 5. The Server Aggregates (FedAvg in Action): The server gathers all the
updated weights from the participating clients. The FedAvg strategy
then works its magic: it calculates a weighted average of these client
models (giving more importance to clients who trained on more data). This averaging process combines the knowledge learned by individual
clients into a new, improved global model. 6. Checking Progress (Server-Side Evaluation): The server now takes
this newly minted global model and evaluates its performance. It uses
the evaluate_fn we provided, testing the model on the global_val_df –
the dataset that no client ever saw during its local training. This gives us
an unbiased look at how well the collaborative model is generalizing. The results (loss, accuracy, recall, etc.) are recorded. 7. Repeat: The process from step 1 (or 2, if thinking of it as a loop) repeats
for the next round, starting with the newly improved global model, until
all FL_NUM_ROUNDS are completed. The Outcome (history):
When all rounds are done, start_simulation gives us back a history object. This object is a treasure trove of information. It contains the performance
metrics (like val_loss, val_acc) logged by our evaluate_fn after each round of
global model evaluation. This history allows us to plot learning curves, see
how the model improved over time, and ultimately compare the performance
of our federated learning system against the centralized baseline. This detailed execution of the simulation, driven by the FedAvg strategy and
95
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
our custom client and evaluation logic, directly demonstrates the federated
averaging algorithm with the ResNet50 V2 model, fulfilling the core objective
of Section 3.5. # 1. Initialize a model instance to get initial parameters for the server
# This model instance is temporary, only for parameter extraction.
initial_model_for_server = LitResNet(
base_model_weights=RESNET_WEIGHTS_CONFIG,
learning_rate=INITIAL_LR_CONFIG, weight_decay=WEIGHT_DECAY_CONFIG, dropout_rate_head_1=0.5, dropout_rate_head_2=0.5,
fine_tune_layers_count=FINE_TUNE_OFFSET_CONFIG, class_weights_tensor=None # Server doesn't train, so class weights not strictly
needed here
)
initial_parameters = ndarrays_to_parameters(
[val.cpu().numpy() for _, val in initial_model_for_server.state_dict().items()]
)
del initial_model_for_server # Clean up
# 2. Define the FedAvg strategy
strategy = FedAvg(
fraction_fit=1.0, # Train on 100% of available clients (i.e., 2 clients)
fraction_evaluate=0.0, # Do not ask clients to evaluate using their
FlowerClient.evaluate method
# We rely on server-side evaluation via `evaluate_fn` min_fit_clients=2, min_evaluate_clients=0, # Consistent with fraction_evaluate=0.0
min_available_clients=2, # Wait for both clients to be available
evaluate_fn=get_evaluate_fn(global_val_df.copy(), IMAGE_DIR), # Server-side
evaluation
initial_parameters=initial_parameters,
96
Project ID: FYP01-DS-T2510-0026
Prepared by: Hamza Khaled CPT6314 Project 2510
)
# 3. Start the simulation
# Ensure client_resources match your setup (GPU or CPU)
client_resources_config = {"num_gpus": 1.0 if DEVICE.type == "cuda" else 0.0, "num_cpus": 1.0}
history = fl.simulation.start_simulation(
client_fn=client_fn, num_clients=2, config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS), strategy=strategy, client_resources=client_resources_config
)