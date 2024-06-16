import yaml
import time
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from utils.data import *
from utils.eval import *
from utils.logging import *
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef
from tensorflow.keras.models import load_model
from models.model import *
from models.model_interface import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dual Language Model based Drug-Target Interactions Prediction"
    )

    parser.add_argument("-c", "--config", help="specify config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["lambda"]["learnable"]:
        lambda_status = "learnable"
    else:
        lambda_status = "fixed-" + str(config["lambda"]["fixed_value"])

    PROJECT_NAME = f"Dataset-{config['dataset']}_ProtT-{config['prot_length']['teacher']}_ProtS-{config['prot_length']['student']}_Lambda-{lambda_status}"
    wandb_logger = WandbLogger(
        name=f"{PROJECT_NAME}", project="DLM_DTI_hint_based_learning"
    )
    print(f"\nProject name: {PROJECT_NAME}\n")

    train_df, valid_df, test_df = load_dataset(mode=config["dataset"])
    print(f"Load Dataset: {config['dataset']}")

    prot_feat_teacher = load_cached_prot_features(
        max_length=config["prot_length"]["teacher"]
    )
    print(
        f"Load Prot teacher's features; Prot Length {config['prot_length']['teacher']}"
    )

    mol_tokenizer, mol_encoder = define_mol_encoder(is_freeze=True)
    prot_tokenizer, prot_encoder = define_prot_encoder(
        max_length=config["prot_length"]["student"],
        hidden_size=config["prot_encoder"]["hidden_size"],
        num_hidden_layer=config["prot_encoder"]["num_hidden_layers"],
        num_attention_heads=config["prot_encoder"]["num_attention_heads"],
        intermediate_size=config["prot_encoder"]["intermediate_size"],
        hidden_act=config["prot_encoder"]["hidden_act"],
    )

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        train_df,
        valid_df,
        test_df,
        prot_feat_teacher=prot_feat_teacher,
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        max_lenght=config["prot_length"]["student"],
        d_mode=config["dataset"],
        batch_size=config["training_config"]["batch_size"],
        num_workers=config["training_config"]["num_workers"],
    )

    model = DTI(
        mol_encoder,
        prot_encoder,
        is_learnable_lambda=config["lambda"]["learnable"],
        fixed_lambda=config["lambda"]["fixed_value"],
        hidden_dim=config["training_config"]["hidden_dim"],
        mol_dim=768,
        prot_dim=config["prot_encoder"]["hidden_size"],
        device_no=config["device"],
    )

    callbacks = define_callbacks(PROJECT_NAME)
    model_interface = DTI_prediction(
        model, len(train_dataloader), config["training_config"]["learning_rate"]
    )
    trainer = pl.Trainer(
        max_epochs=config["training_config"]["epochs"],
        gpus=[config["device"]],
        enable_progress_bar=True,
        callbacks=callbacks,
        precision=16,
        logger=wandb_logger,
    )
    
    model_checkpoint= '/home/resperanca/Projeto/DLM-DTI/weights/Dataset-DAVIS_ProtT-702_ProtS-702_Lambda-learnable/DTI-epoch=045-valid_loss=0.4951-valid_auroc=0.9021-valid_auprc=0.4262.ckpt'
    checkpoint = torch.load(model_checkpoint)
   
    model_interface.load_state_dict(checkpoint["state_dict"])
    
    predictions = trainer.predict(model_interface, test_dataloader)
    results = evaluate(predictions)
    
     # Extraindo as previsões e os rótulos verdadeiros
    y_true = []
    y_pred = []
    for batch in test_dataloader:
        mol_feature, prot_feat_student, prot_feat_teacher, y, source = batch
        y_true.extend(y.cpu().numpy())
        pred_batch, _, _ = model_interface.predict_step(batch, 0)
        y_pred.extend(pred_batch.detach().cpu().numpy())
    
    # Convertendo para arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convertendo previsões de probabilidades para rótulos binários
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculando a matriz de confusão
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
    print= (conf_matrix)


    
    

    
