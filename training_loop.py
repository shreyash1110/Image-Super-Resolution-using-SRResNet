# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0

    for batch_idx, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)

        loss = total_loss(hr, sr, vgg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_psnr += psnr_batch(hr, sr)

        if batch_idx % 200 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Batch {batch_idx+1}/{len(train_dataloader)} → "
                  f"Loss: {loss.item():.4f}")

    # Compute average loss and PSNR for the epoch
    avg_loss = epoch_loss / len(train_dataloader)
    avg_psnr = epoch_psnr / len(train_dataloader)

    loss_epochwise[epoch] = avg_loss
    psnr_epochwise[epoch] = avg_psnr

    print(f"🧪 Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | PSNR = {avg_psnr:.2f} dB")

    # Validation
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for val_lr, val_hr in test_dataloader:
            val_lr = val_lr.to(device)
            val_hr = val_hr.to(device)

            val_sr = model(val_lr)
            v_loss = total_loss(val_hr, val_sr, vgg)

            val_loss += v_loss.item()
            val_psnr += psnr_batch(val_hr, val_sr)

    avg_val_loss = val_loss / len(test_dataloader)
    avg_val_psnr = val_psnr / len(test_dataloader)

    valid_loss_epochwise[epoch] = avg_val_loss
    valid_psnr_epochwise[epoch] = avg_val_psnr

    print(f"✅ Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f} | Val PSNR = {avg_val_psnr:.2f} dB")

    # Step the scheduler
    scheduler.step()

    # Save model checkpoint
    checkpoint_path = f"checkpoints/srresnet_epoch{epoch+1}.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Checkpoint saved to {checkpoint_path}\n")
