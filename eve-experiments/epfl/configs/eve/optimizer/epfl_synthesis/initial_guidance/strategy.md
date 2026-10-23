# Initial synthesis search guidance

The seed expands ABC's resyn2 sequence. Try changes to ordering, zero-cost
rewrites, refactoring or resubstitution while respecting every circuit's delay
cap. Track area and delay separately, and preserve every trial with its recipe.
A lower area on a single circuit does not compensate for failing another circuit.
Use the existing training feedback to choose the next bounded experiment.
