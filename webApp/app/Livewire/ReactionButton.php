<?php
namespace App\Livewire;

use App\Models\Reactions;
use Livewire\Component;

class ReactionButton extends Component
{
    public $threadId;
    public $count;
    public $reacted = false;

    public function mount($threadId)
    {
        $this->threadId = $threadId;

        $this->reacted = Reactions::where('threadId', $threadId)->where('userId', auth()->id())->exists();

        $this->count = Reactions::where('threadId', $threadId)->count();
    }

    public function toggle()
    {
        $reaction = Reactions::where('threadId', $this->threadId)->where('userId', auth()->id())->first();

        if ($reaction) {
            $reaction->delete();
            $this->reacted = false;
            $this->count--;
        } else {
            Reactions::create([
                'threadId' => $this->threadId,
                'userId'   => auth()->id(),
            ]);
            $this->reacted = true;
            $this->count++;
        }
    }

    public function render()
    {
        return view('livewire.reaction-button');
    }
}
