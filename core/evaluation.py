import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from core.factory import build_model
from core.constants import ACTIVITY_MAP

def evaluate_file(pth_path: str, arch: str, dropout: float, dl_test: DataLoader, device: torch.device, args) -> dict:
    if getattr(args, 'task', 'activity') == 'activity':
        num_classes = len(ACTIVITY_MAP)
    elif getattr(args, 'task', 'activity') == 'person_id':
        num_classes = 128
    else:
        num_classes = len(ACTIVITY_MAP)
        
    model = build_model(arch, num_classes, dropout, task=getattr(args, 'task', 'activity')).to(device)
    try:
        loaded = torch.load(pth_path, map_location=device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
    except RuntimeError as e:
        print(f"\nSkipping {pth_path}: Incompatible architecture (likely an older model version).\n{e}")
        return None
        
    model.eval()

    all_preds = []
    all_labels = []
    all_embeddings = []
    debug_samples = []
    total_loss = 0.0
    
    is_person_id = getattr(args, 'task', 'activity') == 'person_id'
    if is_person_id:
        criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    from tqdm import tqdm
    with torch.no_grad():
        basename = pth_path.split('/')[-1]
        for step, batch in enumerate(tqdm(dl_test, desc=f"Testing {basename[:15]}...", leave=False)):
            if hasattr(args, 'fast_dry_run') and args.fast_dry_run and step >= 2:
                break
                
            if is_person_id:
                anchor, positive, negative, labels = batch
                anchor = anchor.to(device).view(anchor.size(0), -1, anchor.size(-1))
                positive = positive.to(device).view(positive.size(0), -1, positive.size(-1))
                negative = negative.to(device).view(negative.size(0), -1, negative.size(-1))
                
                out_a = torch.nn.functional.normalize(model(anchor), p=2, dim=1)
                out_p = torch.nn.functional.normalize(model(positive), p=2, dim=1)
                out_n = torch.nn.functional.normalize(model(negative), p=2, dim=1)
                loss = criterion(out_a, out_p, out_n)
                total_loss += loss.item()
                
                all_embeddings.append(out_a.cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                batch_x, batch_y = batch
                batch_x_device = batch_x.to(device)
                batch_y_device = batch_y.to(device)
                batch_x_view = batch_x_device.view(batch_x_device.size(0), -1, batch_x_device.size(-1))
                
                out = model(batch_x_view)
                loss = criterion(out, batch_y_device)
                total_loss += loss.item()
                preds = out.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                # Collect up to 10 misclassified samples for debug visualization
                if len(debug_samples) < 10:
                    wrong_idx = (preds != batch_y.cpu()).nonzero(as_tuple=True)[0]
                    for idx in wrong_idx:
                        if len(debug_samples) < 10:
                            debug_samples.append({
                                'input': batch_x[idx].cpu().numpy(),
                                'true': batch_y[idx].item(),
                                'pred': preds[idx].item()
                            })

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    if is_person_id:
        # Use 5-NN cross-validation on the test set embeddings to evaluate manifold separation
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_predict
        
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        y_pred = cross_val_predict(knn, all_embeddings, y_true, cv=5, n_jobs=1)
        all_preds = y_pred.tolist()
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    else:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        import wandb
        if wandb.run is not None:
            if getattr(args, 'task', 'activity') == 'activity':
                labels = [k for k, v in sorted(ACTIVITY_MAP.items(), key=lambda item: item[1])]
            elif getattr(args, 'task', 'activity') == 'person_id':
                labels = ["Subject 4", "Subject 5"]
            else:
                labels = ["Subject 4", "Subject 5"]
                
            wandb.log({
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    preds=y_pred, y_true=y_true, class_names=labels,
                    title="Confusion Matrix"
                )
            })
            
            # Log debug samples to W&B
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            wandb_images = []
            for i, sample in enumerate(debug_samples):
                img = sample['input']
                if len(img.shape) == 3 and img.shape[0] == 1:
                    img = img[0]
                elif len(img.shape) == 3:
                    img = img[0]
                    
                fig, ax = plt.subplots()
                im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='jet')
                plt.colorbar(im, ax=ax)
                true_label = labels[sample['true']]
                pred_label = labels[sample['pred']]
                ax.set_title(f"True: {true_label} | Pred: {pred_label}")
                
                fig.canvas.draw()
                image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
                wandb_images.append(wandb.Image(image_array, caption=f"Sample {i+1}"))
                plt.close(fig)
            if wandb_images:
                wandb.log({"Test Misclassifications": wandb_images, "dropout_percent": int(dropout*100)})
    except Exception as e:
        print(f"Error reporting test samples to W&B: {e}")
        # fallback labels just in case
        if getattr(args, 'task', 'activity') == 'activity':
            labels = [k for k, v in sorted(ACTIVITY_MAP.items(), key=lambda item: item[1])]
        elif getattr(args, 'task', 'activity') == 'person_id':
            labels = ["Subject 4", "Subject 5"]
        else:
            labels = ["Subject 6", "Subject 7"]
    
    return {
        "Loss": total_loss / max(1, len(dl_test)),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "labels": labels
    }

def evaluate_network(dataloader: torch.utils.data.DataLoader,
                     model: torch.nn.Module, data_split: str,
                     criterion: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     task_name: str = 'activity') -> None:
    """Run a full evaluation pass and print loss / accuracy / F1."""
    from tqdm import tqdm
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        all_preds, all_labels, total_loss = [], [], 0.0
        debug_samples = []

        for batch_x, batch_y in tqdm(dataloader, desc=f"{data_split} Evaluation"):
            batch_x_device = batch_x.to(device)
            batch_y_device = batch_y.to(device)
            batch_x_view = batch_x_device.view(batch_x_device.size(0), -1, batch_x_device.size(-1))

            y_pred = model(batch_x_view)
            total_loss += criterion(y_pred, batch_y_device).item()
            all_preds.append(y_pred.cpu())
            all_labels.append(batch_y_device.cpu())
            
            # Collect up to 10 misclassified samples for debug visualization
            if len(debug_samples) < 10:
                pred_classes_batch = y_pred.argmax(dim=1).cpu()
                wrong_idx = (pred_classes_batch != batch_y.cpu()).nonzero(as_tuple=True)[0]
                for idx in wrong_idx:
                    if len(debug_samples) < 10:
                        debug_samples.append({
                            'input': batch_x[idx].cpu().numpy(),
                            'true': batch_y[idx].item(),
                            'pred': pred_classes_batch[idx].item()
                        })

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        avg_loss = total_loss / len(dataloader)

        _, pred_classes = torch.max(all_preds, 1)
        true_np = all_labels.numpy()
        pred_np = pred_classes.numpy()

        acc = accuracy_score(true_np, pred_np)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_np, pred_np, average="weighted", zero_division=0)

        print(f"\n{data_split} — loss: {avg_loss:.4f}, acc: {acc:.4f}, "
              f"prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")
              
        try:
            from sklearn.metrics import confusion_matrix
            import wandb
            from core.constants import ACTIVITY_MAP
            if wandb.run is not None:
                if task_name == 'activity':
                    labels = [k for k, v in sorted(ACTIVITY_MAP.items(), key=lambda item: item[1])]
                else:
                    labels = ["Subject 6", "Subject 7"]
                    
                wandb.log({
                    f"Confusion Matrix ({data_split})": wandb.plot.confusion_matrix(
                        preds=pred_np, y_true=true_np, class_names=labels,
                        title=f"Confusion Matrix ({data_split})"
                    )
                })
                
                # Log debug samples to W&B
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                wandb_images = []
                for i, sample in enumerate(debug_samples):
                    img = sample['input']
                    # Squeeze channel dimension if it exists (e.g. 1xTxF -> TxF)
                    if len(img.shape) == 3 and img.shape[0] == 1:
                        img = img[0]
                    elif len(img.shape) == 3:
                        img = img[0] # fallback to first channel
                        
                    fig, ax = plt.subplots()
                    # Transpose to shape (F, T) so time is x-axis, frequency is y-axis
                    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='jet')
                    plt.colorbar(im, ax=ax)
                    true_label = labels[sample['true']]
                    pred_label = labels[sample['pred']]
                    ax.set_title(f"True: {true_label} | Pred: {pred_label}")
                    
                    fig.canvas.draw()
                    image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
                    wandb_images.append(wandb.Image(image_array, caption=f"Sample {i+1}"))
                    plt.close(fig)
                if wandb_images:
                    wandb.log({f"{data_split} Misclassifications": wandb_images})
        except Exception as e:
            print(f"Error reporting to W&B: {e}")
