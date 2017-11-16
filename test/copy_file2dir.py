#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : copy_file2dir.py
# @Author: zjj421
# @Date  : 17-9-11
# @Desc  :
import shutil

import os


def copy_file(src_path, dst_path):
    shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    src_file_list = "006ec59fa59dd80a64c85347eef810c7.aps,038d648c2f29cb0f945c865be25e32e9.aps,08daf3a6cdb4f5e1ad15cb2431ea419a.aps,0cc766de2bb64f0b38488616089a7249.aps,0d925b71485ba1f293ef8abb53fcd141.aps,0fc066d8ab1c5a6a42b636c1fc5876a6.aps,12b3f3c81861b302d87117362b29a02e.aps,131894a9a9203782faac6fa3998621ec.aps,137cd2942a4d022921fda492ff79d40f.aps,16eb63b55c0ab56b543fae2d988bdfcb.aps,1a10297e6c3101af33003e6c4f846f47.aps,1bb32f6a0d5829633a8b0d531aa201e7.aps,1d5dffa877e2d241e39fb155584adca0.aps,23bff5d4daf63345728d4cc6de176aa7.aps,256c90cabcc0d341326430f3e78abcff.aps,2e26af58cf0574ff31a48283eef53d57.aps,340e903fdb6b4d63202fd141638f244d.aps,3775a16c58aed83497088371766a8fb4.aps,3ea9575fe8926bbb01e02b77c0802668.aps,413c6337a5b11c91f24c648c165b3f1a.aps,416a6888eb7ad8c4416fe0b620435136.aps,41ed36bdc93a2ff5519c76f263ab1a88.aps,4275a47cbd350f67c9691d7b52313ac1.aps,471829836c7df1fa0c63721d09ea6db9.aps,4e952b0ef289e292f3fbf82b88538d49.aps,4f4de7a837b9a6122571aa09cc6cda66.aps,58c28b234cc98d3343b090a6b3940bbd.aps,623c761b4db398ea2157e6c5cd6c8c58.aps,65881d8a53837de62f500ceba2d09623.aps,65c89a7cccabe529cc81e0ab9ddea2ce.aps,683a57ebe422fa68ab0065cfd020baa3.aps,68f5497e17d491727d0a27140fbe71cf.aps,761c31e6a7a0a56aaa4345d43ace5add.aps,7a6f34b9aae08fe0c5c5cff245a798c5.aps,7e93c90e40f7d84fc534a20cd1fc23d3.aps,8085d6bf5256a67fd53b545dbb53caf4.aps,97c9d87505f6a2700fb52ba2454266d8.aps,9981e66246cd3351ead1820380a394ee.aps,9f1b80be911f6ec2f5617f9e37a0bbc3.aps,9fbb8962adbe0cdcdea4bed068ab2d18.aps,ab0b1d691aa92456f93c8392a0ec1f9c.aps,b3f223c3c8c3cd6c7f85ade10365917c.aps,b8fc0bcbd1cc95db4d2887fa3c40ac5e.aps,c0c125567e995e4a08a13a0574b61a8c.aps,c139eec534f09d1667c0cd77d05e6495.aps,c335b4799c9e3165f459e424a73409c5.aps,c684f056e0b991d23b27cb42e1005c29.aps,c8fe5162c81b7bc66e802d9ff0bba914.aps,cce55cf0574448bbd16697167f212845.aps,cd39e774d22924b0afbfbb600464b2a8.aps,cfc67388fd826481f71ce414d16b9f16.aps,d27178f5d1d6ac05f9bbe9dc0158d784.aps,e0a5b7b83316cf6dfce4a17304a84651.aps,e4e72e96bffb21e55381667911281616.aps,e75ec32aa94a4d092e6f59c3194aff2a.aps,e938d27a69495739699e989798ccfa5e.aps,e9d689ab1d4af3dec583132948cc2273.aps,e9e2bc4f4f319943935551d502fc11c2.aps,ec9c7903d4665303f7d3150399af8d84.aps,f412f718c4ef81b6a7ce4b46651596ce.aps,f43eaa14eb598d744ebbca0c3cce454b.aps,f5044f179fdbce9ae3f5da283babd359.aps,f859802d4f5cec2b3220725ce54b3322.aps,f92d8566ff9460451ab42093098a0efe.aps,fa0eddae7fa3969578d8216fe1840e4c.aps,fae2676a3d4bd35b0b7088fad9f2e554.aps"
    _, *src_file_list = src_file_list.split(",")
    # print(_)
    # print(src_file_list)
    # print(len(src_file_list))
    src_file_list.append(_)
    # print(src_file_list)
    # print(len(src_file_list))
    src_path_list = list(map(lambda x:os.path.join("/media/zj/study/kaggle/stage1_aps", x), src_file_list))
    src_path_list.sort()
    print(src_path_list)
    # print(len(src_path_list))
    dst_path_list = list(map(lambda x:os.path.join("/media/zj/study/kaggle/need_copy", x), src_file_list))
    dst_path_list.sort()
    print(dst_path_list)
    for s in range(len(src_path_list)):
        if os.path.basename(src_path_list[s]) == os.path.basename(dst_path_list[s]):
            copy_file(src_path_list[s], dst_path_list[s])
        else:
            print("有文件没被复制")
            print(src_path_list[s])
    print("All have done!")