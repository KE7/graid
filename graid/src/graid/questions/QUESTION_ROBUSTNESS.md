### General Strategies for Robustness

Before diving into specific questions, two general strategies can improve robustness across the board:

1.  **Confidence-Based Filtering:** Almost all detector outputs include a confidence score for each bounding box. A simple and highly effective strategy is to only consider detections above a certain confidence threshold (e.g., 0.75). This directly mitigates issues from low-confidence, often incorrect, labels.
2.  **Semantic Grouping:** Create a class hierarchy (e.g., 'car', 'bus', 'truck' -> 'vehicle'; 'cat', 'dog' -> 'animal'). Performing analysis on these parent classes makes the logic robust to misclassifications within a super-category (e.g., mistaking a 'truck' for a 'car' doesn't affect the 'vehicle' count).

---

### 1. `IsObjectCentered` ✅ IMPLEMENTED

*   **Question:** Asks which third of the image a single-instance object is in.
*   **Error Impact:**
    *   **Low Recall:** If one of two `person` objects is missed, the system will incorrectly assume there is only one `person` and ask this question about it.
    *   **Wrong Label:** If a `chair` is mislabeled as a `person`, the question might be asked about the position of the `chair` under the `person` label, leading to a factually incorrect Q&A pair about the scene.

*   **IMPLEMENTED SOLUTIONS:**
    1.  ✅ **Ambiguity Buffer:** Added configurable `buffer_ratio` (default 5% of image width) that creates no-ask zones around the one-third and two-third lines. Questions are skipped if any edge of the bounding box falls within these buffer zones.
    2.  ✅ **Multiple Choice Format:** Converted to multiple choice with clear instructions: "Divide the image into thirds. In which third does the {object_1} primarily appear? Respond with the letter only: A) left third, B) middle third, C) right third."
    3.  ✅ **Letter-Only Answers:** Answers are now just "A", "B", or "C", eliminating ambiguity in response format.

---

### 2. `WidthVsHeight` ✅ IMPLEMENTED

*   **Question:** Asks if a single-instance object is wider than it is tall (or vice-versa).
*   **Error Impact:**
    *   **Low Recall:** Same as `IsObjectCentered`, can lead to mistakenly identifying an object as a single instance.
    *   **Wrong Label:** The question would be about the aspect ratio of a mislabeled object. The geometric answer might be correct for the box, but semantically wrong for the scene.

*   **IMPLEMENTED SOLUTIONS:**
    1.  ✅ **Increased Aspect Ratio Threshold:** Changed from 0.3 to 0.75, so questions are only asked when width is at least 1.75x height (or vice versa), making it robust to minor bounding box noise.
    2.  ✅ **Non-Articulated Classes Filter:** Added `non_articulated_classes` parameter that restricts questions to objects with fixed aspect ratios (e.g., cars, chairs, tables). Excludes pose-variant objects like people and animals.
    3.  ✅ **Clear Yes/No Answers:** Maintained simple "yes"/"no" format for consistency with other questions.

---

### 3. `Quadrants` ✅ IMPLEMENTED

*   **Question:** Asks which quadrant of an N x M grid a single-instance object is in.
*   **Error Impact:**
    *   **Low Recall / Wrong Label:** The same impact as for `IsObjectCentered`.

*   **IMPLEMENTED SOLUTIONS:**
    1.  ✅ **Margin-Based Filtering:** Added configurable `margin_ratio` (default 10% of quadrant size) that requires bounding boxes to be fully contained within quadrants with safety margins. Prevents questions about objects near quadrant boundaries.
    2.  ✅ **Size-Based Filtering:** Only asks questions if the object is small enough to fit within a quadrant with margins, avoiding ambiguous cases with large objects.
    3.  ✅ **Numeric Quadrant Answers:** Maintains clear numeric answers (1, 2, 3, etc.) based on left-to-right, top-to-bottom numbering.

---

### 4. `LargestAppearance` + NEW: `RankLargestK` ✅ IMPLEMENTED

*   **Question:** `LargestAppearance` asks which object class appears largest; `RankLargestK` ranks the top K classes by largest instance.
*   **Error Impact:**
    *   **Low Recall:** Highly sensitive. If the true largest object is not detected, the answer is guaranteed to be wrong.
    *   **Wrong Label:** If the largest object is mislabeled (e.g., a `bus` is labeled as a `car`), the answer will be the incorrect class.

*   **IMPLEMENTED SOLUTIONS:**
    1.  ✅ **New RankLargestK Question:** Created new question class that ranks top K object classes by their largest single instance. Takes `k` parameter and `margin_ratio` for robust ranking.
    2.  ✅ **Margin-Based Ranking:** RankLargestK requires significant area differences between consecutive ranks (default 30% margin) to ensure robust ordering against detection noise.
    3.  ✅ **Comma-Separated List Format:** Answer format is "car, bus, person" eliminating linguistic complexity around ordinals like "first", "second", etc.
    4.  ✅ **Largest-Per-Class Logic:** Both questions now compare the largest instance of each class rather than all individual detections.

---

### 5. `MostAppearance` & 6. `LeastAppearance` ✅ IMPLEMENTED

*   **Question:** Asks which object class appears most or least frequently.
*   **Error Impact:**
    *   **Low Recall:** Very sensitive. Missing a few instances of a class can easily change its rank from most to least frequent.
    *   **Wrong Label:** Very sensitive. A single mislabeling of a `car` to a `truck` affects two counts simultaneously.

*   **IMPLEMENTED SOLUTIONS:**
    1.  ✅ **Margin-Based Count Filtering:** Added configurable `margin_ratio` (default 20%) that requires the winning class count to exceed the runner-up by the specified margin. For MostAppearance: `top_count > (1 + margin) * second_count`.
    2.  ✅ **Robust Count Comparison:** Questions are only asked when there's a clear winner, providing a robustness buffer against small counting errors.
    3.  ✅ **Consistent Answer Format:** Maintains simple class name answers for easy parsing and evaluation.

---

### 7. `LeftOf` & 8. `RightOf`

*   **Question:** Asks if an instance of `{object_1}` is to the left/right of an instance of `{object_2}`.
*   **Error Impact:**
    *   **Low Recall:** Can cause false negatives. If the only `person` to the left of a `tree` is missed, the answer will be incorrectly "No".
    *   **Wrong Label:** Can cause false positives. If a `lamppost` to the left of a `tree` is mislabeled as a `person`, a factually incorrect Q&A will be generated.

*   **Proposed Solutions:**
    1.  **Require Unambiguous Separation:** Strengthen the condition. Instead of just one instance being to the left of another, require that *all* instances of `{object_1}` are to the left of *all* instances of `{object_2}`. This can be checked by verifying `max_x(all_obj1) < min_x(all_obj2)`. This asks the question only in the clearest, most unambiguous scenarios.
    2.  **Aggregate Position:** Base the decision on the average position (centroid) of each class. Ask the question only if the centroid of all `{object_1}` instances is significantly to the left of the centroid of all `{object_2}` instances. This is robust to one or two outliers.
    3.  **Answer with "Sometimes":** If the condition is not absolute (i.e., some `obj1` are left of `obj2`, but some are not), introduce a "Sometimes" or "In some cases" answer. This more accurately reflects complex scenes.

---

### 9. `LeftMost` & 10. `RightMost`

*   **Question:** Asks for the class label of the leftmost/rightmost object.
*   **Error Impact:**
    *   **Low Recall:** Highly sensitive. If the true leftmost object is not detected, the question is being answered about the wrong object, and the answer is guaranteed to be incorrect.
    *   **Wrong Label:** The system correctly identifies the leftmost box, but gives it the wrong label.

*   **Proposed Solutions:**
    1.  **Class-Agnostic Questioning:** Rephrase the question to be about the *properties* of the leftmost object, sidestepping the label issue. For example, "Does the leftmost object appear wider than it is tall?". This is the approach taken by `LeftMostWidthVsHeight` and is very robust to label error.
    2.  **Check for Ambiguity:** Before asking, check if the second-leftmost object is very close to the leftmost one. If their positions are nearly identical, the title of "leftmost" is ambiguous and sensitive to error. In this case, either avoid the question or mention both objects in the answer.
    3.  **"Set-of-Mark" Verification:** As the code comments suggest, this is a prime candidate for Set-of-Mark prompting. Generate the image with all detected boxes drawn on it. Feed this to a VQA model and ask, "What is the label of the object in the leftmost box?". The VQA may be able to correct the detector's label error.

---

### 11. `HowMany` → REPLACED WITH 3 NEW QUESTIONS ✅ IMPLEMENTED

*   **Original Question:** Asked for exact count of a specific object class.
*   **Error Impact:** Direct report of detector output, highly sensitive to both recall and precision errors.

*   **IMPLEMENTED REPLACEMENTS:**

#### 11a. `MoreThanThresholdHowMany` ✅ IMPLEMENTED
*   **Question:** "Are there {target} or more {object_1}(s) in this image? Respond Yes/No."
*   **Robustness:** Uses multiplicative thresholds to create buffer zones. For detected count N, generates two questions:
    - Yes case: target = ⌊N / threshold⌋ (answer: "Yes")  
    - No case: target = ⌈N × threshold⌉ (answer: "No")
*   **Benefits:** Balanced Yes/No distribution, tolerant to counting errors

#### 11b. `LessThanThresholdHowMany` ✅ IMPLEMENTED  
*   **Question:** "Are there less than {target} {object_1}(s) in this image? Respond Yes/No."
*   **Robustness:** Symmetric logic to MoreThanThresholdHowMany with inverse thresholds
*   **Benefits:** Provides complementary threshold-based questions for balanced evaluation

#### 11c. `MultiChoiceHowMany` ✅ IMPLEMENTED
*   **Question:** "How many {object_1}(s) are in the image? Choose one: A) {range_a}, B) {range_b}, C) {range_c}, D) Unsure / Not Visible."
*   **Robustness Features:**
    - Contiguous range buckets based on detected count (low/mid/high)
    - Confidence variance adaptation: wider ranges for uncertain detections
    - Random A/B/C shuffling to prevent positional bias
    - Detector count always falls in middle bucket by design
*   **Benefits:** Tolerates ±1-2 counting errors while maintaining clear boundaries

---

### 12. `AreMore`, 13. `WhichMore` ⚠️ NOT YET IMPLEMENTED

*   **Analysis:** These questions are comparative versions of `HowMany` and suffer from the same sensitivities. The solutions for `MostAppearance` are directly applicable here. The most important is to **require a significant difference in counts** before asking the question to ensure the comparison is robust to minor counting errors.

*   **PLANNED SOLUTIONS:**
    1.  **Margin-Based Count Filtering:** Apply same margin logic as MostAppearance/LeastAppearance
    2.  **Significant Difference Requirement:** Only ask when count differences exceed threshold to avoid tie-breaking scenarios

---

### 14. `LeftMostWidthVsHeight`, 15. `RightMostWidthVsHeight` ⚠️ NOT YET IMPLEMENTED

*   **Analysis:** These are excellent examples of robust question generation. By making the question class-agnostic ("Does the leftmost object..."), they are already immune to **Wrong Label** errors. The primary weakness is **Low Recall** (if the true leftmost object is missed).
*   **PLANNED SOLUTIONS:**
    1.  **Confirm Subject Identity in Answer:** The question can be class-agnostic, but the answer can reveal the label. Q: "Does the leftmost object appear to be wider than it is tall?" A: "Yes. The object, identified as a car, is wider than it is tall." This makes any label error transparent.
    2.  **Ensure Spatial Stability:** Before asking, confirm the identified leftmost object is significantly farther to the left than the next contender. This prevents small box errors from changing the subject of the question.

---

### 16. `ObjectsInRow`, 17. `ObjectsInLine`, 18. `MostClusteredObjects` ✅ IMPLEMENTED

*   **Analysis:** These questions rely on the spatial relationships between multiple objects.
*   **Error Impact:**
    *   **Low Recall:** Missing an object can break a row or cluster. Conversely, missing objects that *aren't* in a row can create the illusion of one among the remaining detections.
    *   **Wrong Label:** This primarily affects `ObjectsInLine` and `MostClusteredObjects`, which report the labels of the grouped objects.

*   **IMPLEMENTED SOLUTIONS:**

#### 16. `ObjectsInRow` ✅ IMPLEMENTED
*   **Linear Regression Approach:** Replaced y-overlap heuristic with linear regression on y-centers. Uses normalized variance threshold (default 0.1 of image height²) for row detection.
*   **Sliding Window Analysis:** Tests multiple window sizes and positions to find the best linear fit.
*   **Robust Yes/No Answers:** Simple binary response format avoiding ambiguous spatial descriptions.

#### 17. `ObjectsInLine` ✅ IMPLEMENTED  
*   **Multiple Choice Format:** "Which objects appear to be arranged in a row? A) {option_a}, B) {option_b}, C) {option_c}, D) No clear row arrangement."
*   **Same Linear Regression Logic:** Uses identical statistical approach as ObjectsInRow for consistency.
*   **Distractor Deduplication:** Implements retry logic (up to 10 attempts) to ensure distractors are unique from correct answer and each other. Skips question if duplicates persist.
*   **Random Shuffling:** Randomizes A/B/C assignment to prevent positional bias.

#### 18. `MostClusteredObjects` ✅ IMPLEMENTED
*   **DBSCAN Clustering:** Replaced distance-based clustering with DBSCAN algorithm for robust cluster detection.
*   **Image-Relative Parameters:** Uses `eps_ratio` (default 5% of image diagonal) instead of fixed pixel distances. Automatically scales for different image sizes.
*   **Increased Requirements:** Now requires ≥9 detections (3 clusters × 3 objects) and `min_samples=3` for meaningful clusters.
*   **Multiple Choice Format:** Same A/B/C structure as ObjectsInLine with distractor deduplication.
*   **Cluster Quality Control:** Requires ≥2 clusters for comparative evaluation and finds most compact cluster using variance-based scoring.

---

## IMPLEMENTATION PROGRESS SUMMARY

### ✅ COMPLETED (11/18 questions)
1. **IsObjectCentered** - Buffer zones, multiple choice A/B/C format
2. **WidthVsHeight** - Increased threshold (0.75), non-articulated classes filter  
3. **Quadrants** - Margin-based filtering (10% quadrant size)
4. **RankLargestK** - NEW: Ranks top-K classes with margin requirements
5. **MostAppearance** - Margin-based count filtering (20%)
6. **LeastAppearance** - Margin-based count filtering (20%)
11. **MoreThanThresholdHowMany** - NEW: Threshold-based Yes/No questions
11. **LessThanThresholdHowMany** - NEW: Inverse threshold questions  
11. **MultiChoiceHowMany** - NEW: 3-way multiple choice with ranges
16. **ObjectsInRow** - Linear regression on y-centers
17. **ObjectsInLine** - Multiple choice with distractor deduplication
18. **MostClusteredObjects** - DBSCAN clustering, image-relative parameters

### ⚠️ PENDING IMPLEMENTATION (7/18 questions)
7. **LeftOf** - No changes planned (already robust)
8. **RightOf** - No changes planned (already robust)  
9. **LeftMost** - No changes planned (already robust)
10. **RightMost** - No changes planned (already robust)
12. **AreMore** - Needs margin-based filtering
13. **WhichMore** - Needs margin-based filtering
14. **LeftMostWidthVsHeight** - Needs spatial stability checks
15. **RightMostWidthVsHeight** - Needs spatial stability checks

### 🎯 KEY ROBUSTNESS IMPROVEMENTS ACHIEVED
- **Buffer Zones:** Prevent questions near spatial boundaries
- **Margin Requirements:** Ensure significant differences before asking questions  
- **Multiple Choice:** Reduce answer ambiguity with A/B/C formats
- **Statistical Methods:** Linear regression for rows, DBSCAN for clusters
- **Image-Relative Parameters:** Scale with image size instead of fixed pixels
- **Distractor Deduplication:** Prevent identical multiple choice options
- **Threshold-Based Questions:** Replace exact counts with range-tolerant questions 

A. LeftMost / RightMost (+ WidthVsHeight variants)
Rival set = any class in our global vocabulary EXCEPT the incumbent leftmost/rightmost class.
A single positive label in RoS → reject question (missed object further out).
B. LeftOf / RightOf
RoS = horizontal band spanning entire height between x_max(obj₁) and x_min(obj₂).
Rival set = object₂’s class for LeftOf (and vice-versa for RightOf).
C. Quadrants / IsObjectCentered
Verify only if bbox is within δ px of a grid boundary.
RoS = margin zone on the opposite side of the claimed quadrant.
D. ObjectsInRow / ObjectsInLine
Rival set = same class labels already in the row.
If SAM finds another instance of those labels within row stripe but outside existing boxes → reject “row” claim.
E. MostClusteredObjects
Check if additional objects of the same candidate class exist inside the cluster centroid radius.
If yes → may reinforce the cluster → keep; if object of other class appears → question remains valid.
Accept/reject based on whether the top-cluster ranking would change.