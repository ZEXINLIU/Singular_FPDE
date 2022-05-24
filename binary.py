import numpy as np

class Solution:

    def ss(self,a):
        print (a+1)
    
    def search(self, nums, target):

        if target not in nums:
            return -1
        else:
            nums.sort()
            mid_ind = int(np.floor(len(nums)/2))
            print (mid_ind, nums)
            while nums[mid_ind] != target:
                if target < nums[mid_ind]:
                    nums = nums[:mid_ind]
                    mid_ind = int(np.floor(len(nums)/2))
                    r = mid_ind
                else:
                    nums = nums[mid_ind:]
                    r = mid_ind + int(np.floor(len(nums)/2))
                
                mid_ind = int(np.floor(len(nums)/2))

            return r

if __name__ == "__main__":
    nums = [-1,0,3,5,9,12]
    target = 0
    s = Solution()
    print (s.search(nums, target))
