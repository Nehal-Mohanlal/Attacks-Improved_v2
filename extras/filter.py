import re 
from collections import defaultdict 
import statistics 


with open('raw_output.txt', 'r', encoding='utf-8') as f: 
    output = f.read() 
    
#print(output) 

#print(f"Basic features found: ", re.findall(r"Feature:\s*(\S+)", output))

pattern = r"Feature:\s*(\S+)\s*changed:\s*\{'from':\s*\d+,\s*'to':\s*(\d+)\}"
    
matches = re.findall(pattern, output, flags=re.DOTALL)

# print(len(matches))

feature_values = defaultdict(list)

for feature, to_val, in matches: 
    feature_values[feature].append(int(to_val))
    
# print(f"feature values: {feature_values}")
print(f"{'Feature':<35} {'Count':>6} {'Avg':>8} {'Min':>6} {'Max':>6} {'Median':>8}") 
print('------------------------------------------------------------------------------')

for feature in feature_values.keys():     
    values = feature_values[feature]
    avg = statistics.mean(values)
    median = statistics.median(values)  
    print(f"{feature:<35} {len(values):>6} {avg:>8.2f} {min(values):>6} {max(values):>6} {median:>8.1f}")