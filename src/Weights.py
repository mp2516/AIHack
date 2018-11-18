def find_flows(edges):
    flows = []
    for edge in edges:
        top = []
        bot = []
        not_done = []
        done = []
        finished = False
        top.append(edge[0])
        bot.append(edge[1])
        count = 0
        while finished == False:
            if count == 0:
                list1 = edges
                count = count + 1
            else:
                list1 = not_done

            for edge2 in list1:
                if edge2 not in done:

                    if edge2 != edge:
                            if edge2[0] in top:
                                if edge2 not in done:
                                    top.append(edge2[1])
                                    done.append(edge2)
                                if edge2 in not_done:
                                    not_done.remove(edge2)
                            if edge2[1] in top:
                                if edge2 not in done:
                                    top.append(edge2[0])
                                    done.append(edge2)
                                if edge2 in not_done:
                                    not_done.remove(edge2)
                            if edge2[0] in bot:
                                if edge2 not in done:
                                    bot.append(edge2[1])
                                    done.append(edge2)
                                if edge2 in not_done:
                                    not_done.remove(edge2)
                            if edge2[1] in bot:
                                if edge2 not in done:
                                    bot.append(edge2[0])
                                    done.append(edge2)
                                if edge2 in not_done:
                                    not_done.remove(edge2)
                            else:
                                if edge2 not in not_done:
                                    not_done.append(edge2)


                    if len(done) + 1 == len(edges):
                        finished = True
        print(edge)
        print(top)
        print(bot)
        weight = 0
        for node in top:
            if node != edge:
                weight = weight + node[2]
        for node in bot:
            if node != edge:
               weight = weight - node[2]

        flows.append(weight/2.0)

    for n in range(len(edges)):
        edges[n].append(flows[n])

    return edges