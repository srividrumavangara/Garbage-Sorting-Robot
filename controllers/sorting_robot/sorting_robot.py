from controller import Supervisor
import sys

# ── Setup ──────────────────────────────────────────────────────────────────
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Class names matching your 5 trained categories
CLASSES = ["wet_organic", "recyclable_plastic", "hazardous", 
           "health_medical", "glass_metal"]

# Colors for each class (RGB) - used to simulate different waste items
CLASS_COLORS = {
    "wet_organic":       [0.4, 0.8, 0.1],
    "recyclable_plastic":[0.1, 0.4, 0.9],
    "hazardous":         [0.9, 0.1, 0.1],
    "health_medical":    [0.9, 0.9, 0.9],
    "glass_metal":       [0.7, 0.7, 0.8],
}

# Bin positions in the simulation (x, z coordinates)
BIN_POSITIONS = {
    "wet_organic":        -1.0,
    "recyclable_plastic": -0.5,
    "hazardous":           0.0,
    "health_medical":      0.5,
    "glass_metal":         1.0,
}

# ── Get camera ─────────────────────────────────────────────────────────────
camera = robot.getDevice("camera")
if camera:
    camera.enable(timestep * 4)
    print("Camera enabled")
else:
    print("No camera found - running in simulation mode")

# ── Get waste item node ────────────────────────────────────────────────────
waste_node = robot.getFromDef("WASTE_ITEM")

# ── Simulation state ───────────────────────────────────────────────────────
current_item_index = 0
step_counter       = 0
STEPS_PER_ITEM     = 60   # steps to wait between items
sort_log           = []

print("\n" + "="*50)
print("GARBAGE SORTING ROBOT - VIRTUAL SIMULATION")
print("="*50)
print(f"Simulating sorting of {len(CLASSES)} waste categories\n")

# ── Main loop ──────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    step_counter += 1

    if step_counter % STEPS_PER_ITEM == 0:
        # Pick the current waste class to simulate
        true_class   = CLASSES[current_item_index % len(CLASSES)]
        
        # Change the waste item color to represent this class
        if waste_node:
            appearance = waste_node.getField("children").getMFNode(0)\
                        .getField("appearance").getSFNode()
            if appearance:
                base_color = appearance.getField("baseColor")
                color      = CLASS_COLORS[true_class]
                base_color.setSFColor(color)

        # ── Simulate model prediction ──────────────────────────────────────
        # In a full deployment this would be:
        #   frame = camera.getImage()
        #   predicted_class, confidence = run_model(frame)
        # For the virtual demo we simulate with realistic accuracy
        # based on your actual trained model's per-class F1 scores
        import random
        accuracy_map = {
            "wet_organic":        0.908,
            "recyclable_plastic": 0.972,
            "hazardous":          0.851,
            "health_medical":     0.863,
            "glass_metal":        0.983,
        }
        correct_prob  = accuracy_map[true_class]
        is_correct    = random.random() < correct_prob
        predicted     = true_class if is_correct else \
                        random.choice([c for c in CLASSES if c != true_class])
        confidence    = round(random.uniform(0.75, 0.99) if is_correct 
                              else random.uniform(0.45, 0.74), 3)

        # ── Confidence-based rejection ─────────────────────────────────────
        CONFIDENCE_THRESHOLD = 0.70
        if confidence < CONFIDENCE_THRESHOLD:
            decision = "MANUAL_REVIEW"
            status   = "REJECTED (low confidence)"
        else:
            decision = predicted
            status   = "CORRECT" if predicted == true_class else "WRONG BIN"

        # ── Move waste item to predicted bin ───────────────────────────────
        if waste_node and decision != "MANUAL_REVIEW":
            pos = waste_node.getField("translation")
            pos.setSFVec3f([BIN_POSITIONS[decision], 0.15, -0.4])
        elif waste_node and decision == "MANUAL_REVIEW":
            pos = waste_node.getField("translation")
            pos.setSFVec3f([0, 0.15, 0.9])  # manual review tray

        # ── Log result ────────────────────────────────────────────────────
        log_entry = {
            "item":       current_item_index + 1,
            "true_class": true_class,
            "predicted":  predicted,
            "confidence": confidence,
            "decision":   decision,
            "correct":    predicted == true_class and decision != "MANUAL_REVIEW"
        }
        sort_log.append(log_entry)

        print(f"Item {log_entry['item']:02d} | "
              f"True: {true_class:<20} | "
              f"Predicted: {predicted:<20} | "
              f"Conf: {confidence:.3f} | "
              f"{status}")

        current_item_index += 1

        # ── Print summary every 25 items ──────────────────────────────────
        if current_item_index % 25 == 0:
            correct   = sum(1 for x in sort_log if x["correct"])
            rejected  = sum(1 for x in sort_log if x["decision"] == "MANUAL_REVIEW")
            total     = len(sort_log)
            attempted = total - rejected
            acc       = correct / attempted if attempted > 0 else 0

            print("\n" + "-"*50)
            print(f"SUMMARY after {total} items:")
            print(f"  Correctly sorted : {correct}/{attempted} ({acc*100:.1f}%)")
            print(f"  Sent for review  : {rejected} ({rejected/total*100:.1f}%)")
            print("-"*50 + "\n")

        # ── Stop after 100 items ──────────────────────────────────────────
        if current_item_index >= 100:
            correct  = sum(1 for x in sort_log if x["correct"])
            rejected = sum(1 for x in sort_log if x["decision"] == "MANUAL_REVIEW")
            total    = len(sort_log)
            attempted = total - rejected
            acc      = correct / attempted if attempted > 0 else 0

            print("\n" + "="*50)
            print("SIMULATION COMPLETE — FINAL RESULTS")
            print("="*50)
            print(f"Total items processed : {total}")
            print(f"Correctly sorted      : {correct} ({correct/total*100:.1f}%)")
            print(f"Sent for manual review: {rejected} ({rejected/total*100:.1f}%)")
            print(f"Sorting accuracy      : {acc*100:.1f}%")
            print("="*50)

            # Save results
            import json, os
            os.makedirs("../../results", exist_ok=True)
            with open("../../results/simulation_results.json", "w") as f:
                json.dump({
                    "total": total,
                    "correct": correct,
                    "rejected": rejected,
                    "accuracy": acc,
                    "log": sort_log
                }, f, indent=2)
            print("\nResults saved to results/simulation_results.json")
            break
