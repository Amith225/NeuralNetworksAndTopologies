# todo: Priority Annotation:
#     #: working on
#     *: work next
#     L: lazy work
#     F: future work
#     D: difficult

# todo: update NEURAL_NETWORKS_AND_ITS_TOPOLOGIES.ipynb file. FL

'''
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch -r \<to delete>' --prune-empty -- --all
git reflog expire --all --expire=now
git gc --prune=now --aggressive
git push origin <your_branch_name> --force

git remote prune origin
git repack
git prune-packed
git reflog expire --expire=now
git gc --aggressive
git gc --prune=now

git filter-repo --analyze
'''
