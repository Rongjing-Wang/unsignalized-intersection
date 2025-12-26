def list_to_dict(in_list):#将一个列表转换为字典
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict): # 将字典的键和值交换
    return dict((in_dict[i], i) for i in in_dict)


def main():
    pass


if __name__ == '__main__':
    main()

