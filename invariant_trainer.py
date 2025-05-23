import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import math
import os
import csv
import numpy as np


import math
import os
from typing import Optional

logger = logging.get_logger(__name__)


class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def remove_dataparallel_wrapper(self):
        if hasattr(self.model, 'module'):
            self.model = self.model.module
    
    def run_evaluation(self):
        """
        Exécute l'évaluation sur self.eval_dataset et renvoie un tuple (eval_loss, perplexity).
        """
        eval_loss = None
        perplexity = None
        if self.eval_dataset is not None:
            eval_metrics = self.evaluate()
            eval_loss = eval_metrics.get("eval_loss")
            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
        return eval_loss, perplexity
    

    def empirical_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            num_train_epochs: Optional[int] = 1,
            **kwargs):
        """
        Fonction d'entraînement pour le fine tuning en mode eLM (empirical risk minimization)
        sur un unique environnement (par exemple, {"all_train": {"train": dataset}}), en intégrant l'utilisation d'AMP.
        
        Pour eLM, la notion de "round" n'est pas nécessaire : l'entraînement se fait par itération séquentielle
        sur le DataLoader (une passe complète sur l'ensemble des batches constitue une époque).
        
        Paramètres :
        - training_set : dictionnaire contenant le dataset d'entraînement, ex. {"all_train": {"train": dataset}}.
        - nb_steps : (optionnel) nombre total de steps ; si précisé, num_train_epochs sera recalculé.
        - num_train_epochs : nombre d'époques à réaliser (défaut 825, pour coller à vos expériences iLM/ensLM).
        - kwargs : arguments supplémentaires (non utilisés ici).
        """

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Au moins nb_steps ou num_train_epochs doit être défini.")
        
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # On suppose qu'il n'y a qu'un seul environnement, par exemple "all_train"
        print(training_set.keys())
        env_key = list(training_set.keys())[0]
        dataset = training_set[env_key]["train"]

        # On obtient le DataLoader pour cet unique environnement
        dataloader = self.get_single_train_dataloader(env_key, dataset)

        # Calcul du nombre de steps par époque (nombre de batches divisés par gradient accumulation)
        num_update_steps_per_epoch = len(dataloader) // self.args.gradient_accumulation_steps
        if nb_steps is not None:
            num_train_epochs = max(1, math.ceil(nb_steps / num_update_steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = num_update_steps_per_epoch * num_train_epochs

        # Création de l'optimiseur et du scheduler pour l'ensemble du modèle
        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model, num_training_steps=max_steps)

        self.state = TrainerState()
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Mise en place du GradScaler pour AMP
        scaler = torch.amp.GradScaler('cuda')

        total_trained_steps = 0
        log_interval = 200  # On log tous les 5 steps
        best_eval_loss = float("inf")

        print("=== Début de l'entraînement eLM (avec AMP) ===")
        print(f"Nombre de batches par époque : {len(dataloader)} (soit environ {num_update_steps_per_epoch} steps d'update)")
        print(f"Nombre total de steps prévus : {max_steps}")

        # Boucle d'entraînement par époque (sans subdivision en rounds)
        for epoch in range(int(num_train_epochs)):
            epoch_loss_sum = 0.0
            epoch_steps = 0
            self.model.train()

            for batch in dataloader:
                if total_trained_steps >= max_steps:
                    break

                optimizer.zero_grad()
                # Déplacement du batch sur le device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                # Forward sous AMP
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()

                # Optionnel : clipping des gradients
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                total_trained_steps += 1
                epoch_loss_sum += loss.item()
                epoch_steps += 1

                if total_trained_steps % log_interval == 0:
                    avg_train_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                    eval_loss, perplexity = self.run_evaluation()
                    print("-" * 50)
                    print(f"Step {total_trained_steps}:")
                    print(f"  Train Loss = {avg_train_loss:.4f}")
                    if eval_loss is not None:
                        print(f"  Eval Loss  = {eval_loss:.4f}")
                    if perplexity is not None:
                        print(f"  Perplexity = {perplexity:.4f}")
                    print("-" * 50)
                    # Sauvegarde dans le CSV des logs d'entraînement
                    if self.is_world_process_zero():
                        log_path = os.path.join(self.args.output_dir, "training_log.csv")
                        # Pour la première écriture, on efface le fichier s'il existe et on écrit l'en-tête
                        if total_trained_steps == log_interval and os.path.exists(log_path):
                            os.remove(log_path)
                        if total_trained_steps == log_interval and not os.path.exists(log_path):
                            with open(log_path, "w") as f:
                                f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                        with open(log_path, "a") as f:
                            f.write(f"{epoch+1},{total_trained_steps},{avg_train_loss:.4f},{eval_loss:.4f},{perplexity:.4f}\n")
                        # Sauvegarde du meilleur modèle si eval_loss est améliorée
                        if eval_loss is not None and eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            best_model_path = os.path.join(self.args.output_dir, "best_model")
                            self.model.save_pretrained(best_model_path, safe_serialization=False)
                            self.model.config.save_pretrained(best_model_path)
                            print(f"--> Nouveau meilleur modèle sauvegardé au step {total_trained_steps} avec eval_loss = {eval_loss:.4f}")
            
            print(f"Fin de l'époque {epoch+1} (steps de cette époque : {epoch_steps})")
            if total_trained_steps >= max_steps:
                break

        print("=== Entraînement eLM terminé. Nombre total de steps :", total_trained_steps)


    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):
        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        for env, data in training_set.items():
            print(f"Environnement {env} : {len(data['train'])} exemples")

        train_lengths = [len(data["train"]) for _, data in training_set.items() if len(data["train"]) > 0]
        if not train_lengths:
            raise ValueError("Tous les jeux d'entraînement sont vides. Vérifie les chemins ou les environnements.")
        min_train_set_size = min(train_lengths)
        print("Taille minimale du jeu d'entraînement :", min_train_set_size)
        
        num_envs = len(training_set)
        # Calcul du nombre d'updates (steps) effectués durant une epoch pour chaque environnement
        num_rounds_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        print("num rounds per epoch", num_rounds_per_epoch)
        print("num envs", num_envs)

        if nb_steps is not None:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            print(f"[DEBUG] nb_steps: {nb_steps}, total_steps_per_epoch: {total_steps_per_epoch}")

            num_train_epochs = max(1, math.ceil(nb_steps / total_steps_per_epoch))
            max_steps = nb_steps
        else:
            total_steps_per_epoch = num_rounds_per_epoch * num_envs
            max_steps = total_steps_per_epoch * num_train_epochs

        # Préparation des DataLoader, optimizers et lr_schedulers pour chaque environnement
        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            
            if hasattr(self.model, "lm_heads"):
                optimizer_env, lr_scheduler_env = self.create_optimizer_and_scheduler(
                    self.model.lm_heads[env_name],
                    num_training_steps=max_steps
                )
            else:
                optimizer_env, lr_scheduler_env = self.create_optimizer_and_scheduler(
                    self.model, num_training_steps=max_steps
                )
            optimizers[env_name] = optimizer_env
            lr_schedulers[env_name] = lr_scheduler_env

        # Optimizer et scheduler pour le modèle partagé (l'encodeur)
        if hasattr(self.model, 'encoder'):
            shared_encoder = self.model.encoder
        elif hasattr(self.model, 'distilbert'):
            shared_encoder = self.model.distilbert
        else:
            raise AttributeError("The model does not have an encoder attribute.")
        
        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(shared_encoder, num_training_steps=max_steps)

        self.state = TrainerState()
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        print("Nombre total d'exemples traités approximativement :", total_train_batch_size * max_steps)
        print(min_train_set_size)

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0
        log_interval = 200  # Par exemple, log tous les 5 steps

        best_eval_loss = float('inf')
        stop_training = False

        # Initialisation du scaler pour AMP
        scaler = torch.amp.GradScaler("cuda")

        # --- Préparation du fichier CSV pour enregistrer l'historique ---
        csv_file = os.path.join(self.args.output_dir, "training_loss_history.csv")
        if self.is_world_process_zero():
            # Si le fichier existe déjà, on le supprime
            if os.path.exists(csv_file):
                os.remove(csv_file)
            # Écriture de l'en-tête
            header = ["Epoch"] + list(training_set.keys())
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        skipped_batches = 0  # 💥 Initialiser le compteur

        for epoch in range(int(num_train_epochs)):
            print("\n" + "=" * 70)
            print(f"===== DÉBUT DE L'ÉPOQUE {epoch + 1}/{num_train_epochs} =====")
            print("=" * 70 + "\n")

            # Dictionnaire pour accumuler la loss de chaque environnement pendant l'époque
            env_epoch_losses = {env_name: [] for env_name in training_set.keys()}

            # Rendre itérables tous les DataLoader par environnement
            iter_loaders = {env_name: iter(dataloaders[env_name]) for env_name in training_set.keys()}

            for round_idx in range(num_rounds_per_epoch):
                round_loss_sum = 0.0
                round_loss_count = 0

                print(f"----- Début du round {round_idx + 1}/{num_rounds_per_epoch} : Mise à jour de tous les environnements -----")
                
                for env_name in training_set.keys():
                    if total_trained_steps >= max_steps:
                        stop_training = True
                        break

                    print(f"[Step {total_trained_steps + 1}] Entraînement sur l'environnement : {env_name}")

                    try:
                        batch = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        batch = next(iter_loaders[env_name])
                    
                    if batch is None:
                        print(f" Batch vide pour {env_name}, on passe au suivant.")
                        continue  # saute ce batch, passe au prochain environnement

                    # On s'assure que la batch est sur le bon device
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    # Réinitialisation des gradients pour l'encodeur partagé et la tête de l'environnement courant
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()
  
                    self.model.train()


                   # Forward avec logits moyennés (toutes les têtes) et AMP
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch, env_name=env_name)
                        
                        loss = outputs.loss

                    # Accumulation de la loss pour le reporting
                    step_loss = loss.item()
                    env_epoch_losses[env_name].append(step_loss)
                    round_loss_sum += step_loss
                    round_loss_count += 1
                    print(f"    Loss pour l'environnement {env_name}: {step_loss:.4f}")

                    # Rétropropagation avec AMP sur encodeur + tête active seulement
                    scaler.scale(loss).backward()


                    # Clipping des gradients si nécessaire
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        scaler.unscale_(optimizers[env_name])
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # Mise à jour des deux optimizers et mise à jour du scaler
                    scaler.step(optimizer)
                    scaler.step(optimizers[env_name])
                    scaler.update()

                    # Mise à jour des schedulers
                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    print(f"    Fin d'update sur l'environnement {env_name}.")

                    total_trained_steps += 1

                    # Sauvegardes éventuelles
                    if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                        self.save_heads(total_trained_steps)
                    if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                        self.save_intermediary_model(total_trained_steps)
                    
                    if stop_training:
                        break
                
                if round_loss_count > 0:
                    avg_round_loss = round_loss_sum / round_loss_count
                else:
                    avg_round_loss = 0.0

                print(f"✅ Round terminé : {skipped_batches} batchs sautés pour cause de batch vide ou NaN/Inf.")
                print(f"----- Fin du round {round_idx + 1}/{num_rounds_per_epoch}: Loss moyenne sur ce round = {avg_round_loss:.4f} -----\n")

                # Logging et évaluation périodique
                if total_trained_steps % log_interval == 0:
                    eval_loss, perplexity = self.run_evaluation()
                    if self.is_world_process_zero():
                        log_path = os.path.join(self.args.output_dir, "training_log.csv")
                        if total_trained_steps == log_interval and os.path.exists(log_path):
                            os.remove(log_path)
                        if total_trained_steps == log_interval and not os.path.exists(log_path):
                            with open(log_path, "w") as f:
                                f.write("epoch,global_step,train_loss,val_loss,perplexity\n")
                        with open(log_path, "a") as f:
                            f.write(f"{epoch + 1},{total_trained_steps},{avg_round_loss:.4f},{eval_loss:.4f},{perplexity:.4f}\n")
                        print(f"--> Résumé [Step {total_trained_steps}] : Loss moyenne = {avg_round_loss:.4f}, Eval Loss = {eval_loss:.4f}, Perplexity = {perplexity:.4f}")
                        

                    if eval_loss is not None and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_path = os.path.join(self.args.output_dir, "best_model")
                        self.model.save_pretrained(best_model_path, safe_serialization=False)
                        print(f"Meilleur modèle sauvegardé à l'étape {total_trained_steps} avec eval_loss = {eval_loss:.4f}")
      
                if stop_training:
                    break

            print(f"===== Fin de l'ÉPOQUE {epoch + 1}/{num_train_epochs} =====\n")
            print("Résumé des pertes moyennes par environnement pour cette époque :")
            # Calcul et affichage des pertes moyennes pour chaque environnement
            epoch_summary = {}
            for env_name, losses in env_epoch_losses.items():
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    epoch_summary[env_name] = avg_loss
                    print(f"  {env_name} : {avg_loss:.4f} (basé sur {len(losses)} updates)")
                else:
                    epoch_summary[env_name] = None
                    print(f"  {env_name} : aucune donnée de loss enregistrée.")

            # Sauvegarde dans un fichier CSV pour pouvoir tracer les courbes ultérieurement
            if self.is_world_process_zero():
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [epoch + 1]
                    for env_name in training_set.keys():
                        row.append(epoch_summary.get(env_name) if epoch_summary.get(env_name) is not None else "")
                    writer.writerow(row)
            
            if stop_training:
                break
        
        print("Entraînement terminé. Nombre total de steps:", total_trained_steps)


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        # Ne sauvegarder que si ce processus est le principal
        if not self.is_world_process_zero():
            return
    
        if not hasattr(self.model, "lm_heads"):
            # Si le modèle n'a pas d'attribut lm_heads (mode eLM), on ne sauvegarde rien.
            return
        
        print("saving-heads")
        if not os.path.exists("lm_heads"):
            os.makedirs("lm_heads")

        for env, lm_head in self.model.lm_heads.items():
            filepath = os.path.join("lm_heads", "{}-{}".format(env, step_count))
            
            if hasattr(lm_head, "dense"):
                np.save(filepath, lm_head.dense.weight.data.cpu().numpy())
            elif hasattr(lm_head, "decoder"):
                np.save(filepath, lm_head.decoder.weight.data.cpu().numpy())
            elif hasattr(lm_head, "vocab_projector"):
                np.save(filepath, lm_head.vocab_projector.weight.data.cpu().numpy())
            else:
                print(f"La tête pour l'environnement {env} ne possède pas d'attribut de sauvegarde connu.")


    def get_single_train_dataloader(self, env_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )