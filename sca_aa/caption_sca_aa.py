import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import torch.backends.cudnn as cudnn
import imageio
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption(encoder, decoder, image_path, word_map, beam_size=5):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    """
    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    image = torch.unsqueeze(image, 0)  # (1, 3, 256, 256)
    
    # Encode
    enc_image, global_features = encoder(image)  # (1, num_pixels, features)
    num_pixels = enc_image.size(1)
    encoder_dim = enc_image.size(2)

    # We'll treat the problem as having a batch size of k
    enc_image = enc_image.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(enc_image)
    spatial_image = F.relu(decoder.encoded_to_hidden(enc_image))  # (k,num_pixels,hidden_size)
        
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (k,embed_dim)
        b_c = decoder.channel_attention(global_features, h)
        global_tmp = b_c * global_features
        global_tmp = F.relu(decoder.global_features(global_tmp))
        inputs = torch.cat((embeddings, global_tmp.expand_as(embeddings)), dim = 1)    
        h, c, st = decoder.LSTM(inputs , (h, c))  # (batch_size_t, hidden_size)
        # Run the adaptive attention model
        out_l, _, _ = decoder.adaptive_attention(spatial_image, h, st)
        # Compute the probability over the vocabulary
        scores = decoder.fc(out_l)      # (batch_size, vocab_size)
        scores = F.log_softmax(scores, dim=1)   # (s, vocab_size)
        # (k,1) will be (k,vocab_size), then (k,vocab_size) + (s,vocab_size) --> (s, vocab_size)
        scores = top_k_scores.expand_as(scores) + scores  
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        # prev_word_inds = top_k_words // vocab_size  # (s)
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='trunc')
        next_word_inds = top_k_words % vocab_size  # (s)
        
        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        spatial_image = spatial_image[prev_word_inds[incomplete_inds]]
        global_features = global_features[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    words = [rev_word_map[ind] for ind in seq]
    print(' '.join(words[1:-1]))

'''
def visualize_att(image_path, seq, alphas, betas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    # image = image.resize([7 * 7, 7 * 7], Image.LANCZOS)
    img = np.array(Image.fromarray(img).resize((49, 48), Image.LANCZOS))
    words = [rev_word_map[ind] for ind in seq]
    print(' '.join(words[1:-1]))

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.text(10, 65, '%.2f' % (1-(betas[t].item())), color='green', backgroundcolor='white', fontsize=15)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=7, sigma=7)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [7 * 7, 7 * 7])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap('jet')
        plt.axis('off')
        
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combining SCA-CNN and Adaptive Attention - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas, betas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, betas, rev_word_map, args.smooth)
'''

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_sca_aa.pth.tar'  # model checkpoint
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
word_map_file = '/code/SCA-AA/input_files/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
caption(encoder, decoder, '/code/SCA-AA/horse.jpeg', word_map_file)
caption(encoder, decoder, '/code/SCA-AA/test.jpg', word_map_file)
